# LiveAvatar 训练流水线补充说明

## 修改思路
- **保持推理-训练结构一致**：直接复用现有推理的 `WanS2V` 模块，开启 `is_training=True`，沿用 FlowMatch 调度与 VAE 编码方式，避免额外实现一套模型。  
- **最小依赖的训练脚手架**：新增 `train_pipeline.py`，包含数据读取、VAE 编码、音频嵌入、LoRA 参数解冻、优化与 checkpoint 保存，便于在现有环境快速启动训练。  
- **数据灵活性**：数据集支持 `audio` 可选；当 `use_audio=False` 时自动填充静音张量，允许视频-only 训练或消融。  
- **批量文本支持**：`encode_prompt` 接收列表输入，训练与推理接口对齐，方便批处理。

## 主要改动
- 新增 `train_pipeline.py`：实现数据管线、分布式采样、FlowMatch 训练循环、LoRA 优化与 checkpoint 落盘。配置通过 YAML/CLI 传入，需提供 `dataset_path` 与 `dataset` 基本参数。  
- 更新 `encode_prompt`（常规/TPP 管线）：支持批量 prompt 列表，保证训练时的批处理兼容。  
- 数据集增强：`AvatarDataset` 支持无音频样本（填充静音），并在 `use_audio=True` 时对缺失音频显式报错，确保语音驱动训练的完整性。

## 可行性原因
- 复用已验证的推理组件（VAE、音频编码、扩散骨干与调度器），减少实现偏差。  
- FlowMatch 训练目标与现有采样器一致，损失为加权 MSE，配合 LoRA 微调可在有限资源下对齐论文设置。  
- 数据格式要求简单（视频帧、可选音频、文本），可通过 JSON/JSONL 列表统一描述，便于扩展与分布式加载。  
- 批量 prompt 与静音填充提升训练鲁棒性，支持带/不带音频的多场景训练实验。
