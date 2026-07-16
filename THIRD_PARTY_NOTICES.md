# 第三方代码声明

本仓库包含并修改了以下开源项目的部分代码：

## VAR

- 项目：Visual AutoRegressive Modeling（VAR）
- 上游仓库：https://github.com/FoundationVision/VAR
- 版权所有：Copyright (c) 2024 FoundationVision
- 许可证：MIT License

## Infinity

- 项目：Infinity: Scaling Bitwise AutoRegressive Modeling for High-Resolution Image Synthesis
- 上游仓库：https://github.com/FoundationVision/Infinity
- 版权所有：Copyright (c) 2024 FoundationVision
- 许可证：MIT License

根目录的 [LICENSE](LICENSE) 保留了上游 MIT License 和版权声明。SAR 相关修改主要涉及最后尺度的多步解码、上下文感知位置选择、匹配的训练与推理入口，以及公开发布所需的可移植性整理。上游代码不被声明为 SAR 作者的原创代码。

## 上游代码中保留的来源引用

复制的 FoundationVision 文件还保留了其原有的逐文件来源注释，包括：

- `var/models/helpers.py` 中来自 [pytorch-image-models（timm）](https://github.com/huggingface/pytorch-image-models) 的 DropPath 实现；该项目代码采用 Apache-2.0，版权归 Ross Wightman 及贡献者所有。
- VAR 的 VQ/VAE 文件引用 [CompVis/taming-transformers](https://github.com/CompVis/taming-transformers)（MIT）以及 [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)；相关来源链接保留在文件头。
- VAR/Infinity 的分布式工具引用 [OpenMMLab MMCV](https://github.com/open-mmlab/mmcv)（Apache-2.0）。
- Infinity 的 VAE 文件引用 [Black Forest Labs FLUX](https://github.com/black-forest-labs/flux) 的 autoencoder 实现；FLUX 代码仓库采用 Apache-2.0。模型权重可能使用不同许可证，本仓库不分发任何 FLUX 或 Infinity 权重。

Apache License 2.0 的副本见 [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt)。任何由用户另外下载的模型 checkpoint、文本编码器和数据集均受各自许可证约束，不因本仓库的代码许可证而改变。
