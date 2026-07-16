import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from huggingface_hub import PyTorchModelHubMixin

import dist
from models.basic_var import AdaLNBeforeHead, AdaLNSelfAttn
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from models.scheduling import nearest_to_anchor, random_unfixed
from models.vqvae import VQVAE, VectorQuantizer2


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)   # B16C


class VAR(nn.Module):
    def __init__(
        self, vae_local: VQVAE,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
        num_gen_steps: int = 1,  # 1=original, 2/4/8=few-step generation
    ):
        super().__init__()
        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads

        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # progressive training

        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2

        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())

        # Few-step generation parameters
        self.num_gen_steps = num_gen_steps
        self.last_scale_tokens = self.patch_nums[-1] ** 2  # 256 for 16x16
        
        # 1. input (word) embedding
        quant: VectorQuantizer2 = vae_local.quantize
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)
        self.word_embed = nn.Linear(self.Cvae, self.C)

        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # 3. absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )
        
        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())

        # 5.5 attention mask for few-step training (full attention on last scale)
        #     前 9 scales: causal (same as original)
        #     最后 scale (256 tokens): full attention (can see all 256 + previous scales)
        if num_gen_steps > 1:
            attn_bias_fewstep = attn_bias_for_masking.clone()
            last_scale_start = self.L - self.last_scale_tokens  # 680 - 256 = 424
            # Last 256 positions can attend to all positions (including each other)
            attn_bias_fewstep[:, :, last_scale_start:, :] = 0.0
            self.register_buffer(
                'attn_bias_fewstep', attn_bias_fewstep.contiguous(), persistent=False
            )
        else:
            self.attn_bias_fewstep = None

        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)
    
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()
    
    @torch.no_grad()
    def autoregressive_infer_cfg(
        self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False,
    ) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        
        if label_B is None:
            label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)
        
        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        
        for b in self.blocks: b.attn.kv_caching(True)
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment
            ratio = si / self.num_stages_minus_1
            # last_L = cur_L
            cur_L += pn*pn
            # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            AdaLNSelfAttn.forward
            for b in self.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
            logits_BlV = self.get_logits(x, cond_BD)
            
            t = cfg * ratio
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]
            
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            if not more_smooth: # this is the default case
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
            else:   # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
            if si != self.num_stages_minus_1:   # prepare for next stage
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
        
        for b in self.blocks: b.attn.kv_caching(False)
        return self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]

    @torch.no_grad()
    def autoregressive_infer_cfg_iterative(
        self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False,
        num_iterations: int = 4,
        order: str = "sar",
    ) -> torch.Tensor:
        """
        Four-step last-scale generation used in the paper.

        Each iteration:
        1. Run full attention (all positions can see each other)
        2. Fix random positions or SAR-selected positions
        3. Next iteration uses fixed tokens as real values, others regenerate

        :param num_iterations: number of refinement iterations (fixes 256/num_iterations tokens per iteration)
        :param order: ``random`` for the retraining baseline or ``sar`` for
            positions nearest to the class token in transformer space.
        """
        assert 256 % num_iterations == 0, f"num_iterations must divide 256 evenly, got {num_iterations}"
        if order not in {"random", "sar"}:
            raise ValueError(f"order must be 'random' or 'sar', got {order!r}")
        tokens_per_iter = 256 // num_iterations

        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng

        if label_B is None:
            label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)

        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))

        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]

        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])

        # Generate first (N-1) scales using KV cache (same as original VAR).

        for b in self.blocks: b.attn.kv_caching(True)

        for si, pn in enumerate(self.patch_nums[:-1]):
            ratio = si / self.num_stages_minus_1
            cur_L += pn*pn
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            for b in self.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
            logits_BlV = self.get_logits(x, cond_BD)

            t = cfg * ratio
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]

            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            h_BLCvae = self.vae_quant_proxy[0].embedding(idx_Bl)  # (B, pn*pn, Cvae)
            h_BChw = h_BLCvae.transpose(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)

            next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
            next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
            next_token_map = next_token_map.repeat(2, 1, 1)

        # Save KV cache state
        prev_cached_k = [b.attn.cached_k.clone() for b in self.blocks]
        prev_cached_v = [b.attn.cached_v.clone() for b in self.blocks]
        for b in self.blocks: b.attn.kv_caching(False)

        # Last scale with iterative refinement
        si = self.num_stages_minus_1
        pn = self.patch_nums[-1]
        ratio = 1.0
        num_tokens = pn * pn  # 256

        # Initialize: all positions use f_hat proxy, none fixed
        fixed_mask = torch.zeros(B, num_tokens, dtype=torch.bool, device=f_hat.device)
        fixed_idx = torch.zeros(B, num_tokens, dtype=torch.long, device=f_hat.device)
        fixed_h = torch.zeros(B, num_tokens, self.Cvae, device=f_hat.device)

        last_scale_pos = lvl_pos[:, cur_L:cur_L + num_tokens]
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)

        for iter_i in range(num_iterations):
            # Build input: fixed positions use real tokens, others use f_hat proxy
            last_scale_input = next_token_map[:B].clone()
            if iter_i > 0:
                fixed_emb = self.word_embed(fixed_h) + last_scale_pos
                last_scale_input = torch.where(
                    fixed_mask.unsqueeze(-1),
                    fixed_emb,
                    last_scale_input
                )

            last_scale_input_cfg = last_scale_input.repeat(2, 1, 1)

            # Full-connected attention (no causal mask for last scale positions)
            # Only need to restore KV cache for previous scales
            for b_idx, b in enumerate(self.blocks):
                b.attn.caching = True
                b.attn.cached_k = prev_cached_k[b_idx].clone()
                b.attn.cached_v = prev_cached_v[b_idx].clone()

            x = last_scale_input_cfg
            for b in self.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)  # No attention bias = full attention
            logits_BlV = self.get_logits(x, cond_BD)

            t = cfg * ratio
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]  # (B, 256, V)

            # Sample all positions
            all_idx = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            all_h = self.vae_quant_proxy[0].embedding(all_idx)

            # Select the next 64 positions. SAR uses transformer-space L2
            # distance to the class token; the training baseline is random.
            if order == "sar":
                top_indices = nearest_to_anchor(
                    x[:B], fixed_mask, cond_BD[:B], tokens_per_iter
                )
            else:
                top_indices = random_unfixed(fixed_mask, tokens_per_iter, rng)

            # Fix selected positions
            for b_idx in range(B):
                positions = top_indices[b_idx]
                fixed_mask[b_idx, positions] = True
                fixed_idx[b_idx, positions] = all_idx[b_idx, positions]
                fixed_h[b_idx, positions] = all_h[b_idx, positions]

        # Clear KV cache
        for b in self.blocks: b.attn.kv_caching(False)

        # Assemble final result
        h_BChw = fixed_h.view(B, pn, pn, self.Cvae).permute(0, 3, 1, 2)
        f_hat, _ = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)

        return self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)

    def forward_multistep_gt_only(
        self,
        label_B: torch.LongTensor,
        x_BLCv_wo_first_l: torch.Tensor,
        gt_BL: torch.LongTensor,
        num_steps: int = 4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-step training with GT only (方案 A - 改进版).

        训练时未固定位置使用 f_hat proxy（和推理一致），不使用 mask_token。
        固定位置的选择保持随机。

        Loss 计算方式 (MaskGIT 风格):
        - 前 9 scales: 只在 Step 0 计算一次
        - 最后 scale: 每步计算未固定位置的 CE loss，然后取平均

        总 Loss = Loss_前9scales + mean(Loss_step_0, ..., Loss_step_N-1)

        Args:
            label_B: (B,) class labels
            x_BLCv_wo_first_l: (B, L-first_l, Cvae) teacher forcing input (VAE embeddings)
            gt_BL: (B, L) ground truth token indices for all positions
            num_steps: number of generation steps for last scale (2, 4, or 8)

        Returns:
            total_loss: scalar, the combined loss
            logits_BLV: (B, L, V) logits from the last step (for logging/debugging)
        """
        import torch.nn.functional as F

        assert 256 % num_steps == 0, f"num_steps must divide 256, got {num_steps}"

        tokens_per_step = self.last_scale_tokens // num_steps  # 256 // num_steps
        B = x_BLCv_wo_first_l.shape[0]
        device = x_BLCv_wo_first_l.device

        # ===== 准备前 9 scales 的输入 =====
        # 必须在 autocast(enabled=False) 内构建，和原始 forward() 保持一致，确保 float32 精度
        with torch.cuda.amp.autocast(enabled=False):
            label_B_dropped = torch.where(
                torch.rand(B, device=device) < self.cond_drop_rate,
                self.num_classes, label_B
            )
            sos = cond_BD = self.class_emb(label_B_dropped)
            sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)

            prev_L = self.L - self.last_scale_tokens  # 424
            x_prev_scales = x_BLCv_wo_first_l[:, :prev_L - self.first_l]  # (B, 423, Cvae)
            x_prev_BLC = torch.cat((sos, self.word_embed(x_prev_scales.float())), dim=1)  # (B, 424, C)
            x_prev_BLC += self.lvl_embed(self.lvl_1L[:, :prev_L].expand(B, -1)) + self.pos_1LC[:, :prev_L]

        # GT for different parts
        gt_prev_scales = gt_BL[:, :prev_L]  # (B, 424)
        gt_last_scale = gt_BL[:, prev_L:]   # (B, 256)
        # GT embedding for last scale: look up from codebook directly
        gt_last_scale_emb = self.vae_quant_proxy[0].embedding(gt_last_scale)  # (B, 256, Cvae)

        # ===== 构建 f_hat proxy（和推理时一致）=====
        # 直接从 gt_BL 的 token indices 查 embedding，和推理时完全一致
        SN = len(self.patch_nums)
        HW = self.patch_nums[-1]  # 16
        f_hat = gt_BL.new_zeros(B, self.Cvae, HW, HW, dtype=torch.float32)

        cur_pos = 0
        for si, pn in enumerate(self.patch_nums[:-1]):
            scale_len = pn * pn
            # 从 gt_BL 取当前 scale 的 token indices，查 codebook 得到 GT embedding
            idx_Bl = gt_BL[:, cur_pos:cur_pos + scale_len]  # (B, pn*pn)
            h_BLCvae = self.vae_quant_proxy[0].embedding(idx_Bl)  # (B, pn*pn, Cvae)
            h_BChw = h_BLCvae.transpose(1, 2).reshape(B, self.Cvae, pn, pn)

            h_upsampled = F.interpolate(h_BChw.float(), size=(HW, HW), mode='bicubic')
            f_hat = f_hat + self.vae_quant_proxy[0].quant_resi[si / (SN - 1)](h_upsampled)

            cur_pos += scale_len

        # 从 f_hat 得到最后 scale 的初始 embedding（和推理时一致）
        # f_hat: (B, Cvae, 16, 16) -> (B, 256, Cvae)
        f_hat_last_scale = f_hat.view(B, self.Cvae, -1).transpose(1, 2)  # (B, 256, Cvae)

        # 获取 dtype
        temp = x_prev_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        cond_BD_or_gss = self.shared_ada_lin(cond_BD).to(dtype=main_type)

        # ===== Multi-step training =====
        fixed_mask = torch.zeros(B, self.last_scale_tokens, dtype=torch.bool, device=device)

        loss_prev_scales = None  # 只算一次
        losses_last_scale = []   # 每步的 loss
        logits_last = None

        for step_i in range(num_steps):
            # ------ 1. 构建最后 scale 的输入 ------
            # Clone fixed_mask before use to avoid inplace modification breaking gradient checkpointing recompute
            fixed_mask_step = fixed_mask.clone()

            # 未固定位置用 f_hat proxy（和推理一致），固定位置用 GT embedding
            last_scale_input_vae = torch.where(
                fixed_mask_step.unsqueeze(-1),
                gt_last_scale_emb,      # 固定位置用 GT embedding
                f_hat_last_scale        # 未固定位置用 f_hat proxy（不是 mask_token）
            )

            last_scale_input = self.word_embed(last_scale_input_vae.float())
            last_scale_input += self.lvl_embed(self.lvl_1L[:, prev_L:].expand(B, -1)) + self.pos_1LC[:, prev_L:]

            x_BLC = torch.cat([x_prev_BLC, last_scale_input], dim=1)

            # ------ 2. Forward ------
            x_BLC = x_BLC.to(dtype=main_type)
            attn_bias = self.attn_bias_fewstep.to(dtype=main_type)

            x = x_BLC
            # Use gradient checkpointing to save memory (required for 4-step training)
            for b in self.blocks:
                if self.training:
                    x = torch.utils.checkpoint.checkpoint(b, x, cond_BD_or_gss, attn_bias, use_reentrant=False)
                else:
                    x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
            logits_BLV = self.get_logits(x.float(), cond_BD)  # (B, L, V)
            logits_last = logits_BLV

            # ------ 3. 计算 Loss ------
            # 前 9 scales 的 loss（只在 Step 0 计算）
            if step_i == 0:
                logits_prev = logits_BLV[:, :prev_L, :].reshape(-1, self.V)  # (B*424, V)
                target_prev = gt_prev_scales.reshape(-1)  # (B*424,)
                loss_prev_scales = F.cross_entropy(logits_prev, target_prev, reduction='mean')

            # 最后 scale 的 loss（只在未固定位置计算）
            unfixed_mask = ~fixed_mask_step  # (B, 256)
            logits_last_scale = logits_BLV[:, prev_L:, :]  # (B, 256, V)

            # 展平并选择未固定位置
            logits_unfixed = logits_last_scale[unfixed_mask]  # (num_unfixed, V)
            target_unfixed = gt_last_scale[unfixed_mask]      # (num_unfixed,)

            if logits_unfixed.numel() > 0:
                loss_step = F.cross_entropy(logits_unfixed, target_unfixed, reduction='mean')
                losses_last_scale.append(loss_step)

            # ------ 4. 随机选择要固定的位置 ------
            if step_i < num_steps - 1:
                positions = random_unfixed(fixed_mask, tokens_per_step)
                fixed_mask.scatter_(1, positions, True)

        # ===== 计算总 Loss =====
        # Loss = Loss_前9scales + mean(Loss_steps)
        loss_last_scale = torch.stack(losses_last_scale).mean()
        total_loss = loss_prev_scales + loss_last_scale

        return total_loss, logits_last

    def forward(self, label_B: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor) -> torch.Tensor:  # returns logits_BLV
        """
        :param label_B: label_B
        :param x_BLCv_wo_first_l: teacher forcing input (B, self.L-self.first_l, self.Cvae)
        :return: logits BLV, V is vocab_size
        """
        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)
        B = x_BLCv_wo_first_l.shape[0]
        with torch.cuda.amp.autocast(enabled=False):
            label_B = torch.where(torch.rand(B, device=label_B.device) < self.cond_drop_rate, self.num_classes, label_B)
            sos = cond_BD = self.class_emb(label_B)
            sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)
            
            if self.prog_si == 0: x_BLC = sos
            else: x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)
            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed] # lvl: BLC;  pos: 1LC
        
        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)
        
        # hack: get the dtype if mixed precision is used
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        
        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)
        
        AdaLNSelfAttn.forward
        for i, b in enumerate(self.blocks):
            x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
        x_BLC = self.get_logits(x_BLC.float(), cond_BD)
        
        if self.prog_si == 0:
            if isinstance(self.word_embed, nn.Linear):
                x_BLC[0, 0, 0] += self.word_embed.weight[0, 0] * 0 + self.word_embed.bias[0] * 0
            else:
                s = 0
                for p in self.word_embed.parameters():
                    if p.requires_grad:
                        s += p.view(-1)[0] * 0
                x_BLC[0, 0, 0] += s
        return x_BLC    # logits BLV, V is vocab_size
    
    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        if init_std < 0: init_std = (1 / self.C / 3) ** 0.5     # init_std < 0: automated
        
        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0: nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else: nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()
        
        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()
        
        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()
        
        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2*self.C:].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[:2*self.C].mul_(init_adaln_gamma)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)
    
    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'


class VARHF(VAR, PyTorchModelHubMixin):
            # repo_url="https://github.com/FoundationVision/VAR",
            # tags=["image-generation"]):
    def __init__(
        self,
        vae_kwargs,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
        num_gen_steps: int = 1,  # 1=original, 2/4/8=few-step generation
    ):
        vae_local = VQVAE(**vae_kwargs)
        super().__init__(
            vae_local=vae_local,
            num_classes=num_classes, depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            norm_eps=norm_eps, shared_aln=shared_aln, cond_drop_rate=cond_drop_rate,
            attn_l2_norm=attn_l2_norm,
            patch_nums=patch_nums,
            flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            num_gen_steps=num_gen_steps,
        )
