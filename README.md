# 基于预训练模型的文本分类与可解释性分析

本项目实现了基于DistilBERT的文本分类任务，使用LoRA进行参数高效微调，并提供注意力可视化、SHAP和LIME等可解释性分析工具。

## 项目特点

- **轻量级模型**: 使用DistilBERT，参数量远小于BERT-base，训练和推理速度更快
- **参数高效微调**: 采用LoRA技术，只需训练约1.3%的参数即可达到优秀效果
- **GPU加速训练**: 支持CUDA加速，训练速度提升10-50倍
- **本地数据支持**: 支持从本地CSV文件加载数据集，无需网络连接
- **完整评估**: 提供准确率、F1分数、精确率、召回率等评估指标
- **可解释性分析**: 集成注意力可视化、SHAP和LIME等多种可解释性工具
- **自动保存**: 训练完成后自动保存最佳模型和最终模型
- **数据增强**: 支持同义词替换、随机插入/删除/交换等多种数据增强技术
- **模型集成**: 支持投票集成、平均概率集成和加权集成等多种集成策略
- **情感相关预处理**: 保留情感相关标点符号，提升模型情感理解能力

## 项目性能

在IMDb电影评论情感分析任务上的表现：

| 指标 | 基础模型 | 增强模型 |
|------|----------|----------|
| **准确率 (Accuracy)** | 90.10% | **94.50%** |
| **F1分数** | 90.10% | **94.50%** |
| **精确率 (Precision)** | 90.10% | **94.50%** |
| **召回率 (Recall)** | 90.10% | **94.50%** |

**训练配置** (增强模型): 
- LoRA rank=32, alpha=64, 扩展目标模块
- 数据增强: 同义词替换、随机插入/删除/交换，增强率=0.3
- batch_size=64, max_length=512, epochs=3

## 项目结构

```
distilbert/
├── pytorch_model.bin      # 预训练模型权重（本地）
├── IMDB/                  # 本地数据集目录
│   ├── Train.csv          # 训练集
│   ├── Test.csv           # 测试集
│   └── Valid.csv          # 验证集
├── config.py              # 配置文件
├── data_loader.py         # 数据加载和预处理
├── model_utils.py         # 模型工具函数（LoRA配置等）
├── train.py               # 训练脚本
├── evaluate.py            # 评估脚本
├── explainability.py      # 可解释性分析脚本
├── ensemble.py            # 模型集成模块
├── main.py                # 主运行脚本
├── requirements.txt       # 依赖包列表
├── saved_models/          # 保存的模型
│   ├── best_model/        # 最佳模型（验证集F1最高）
│   ├── final_model/       # 最终模型（训练完成时）
│   └── training_info.json # 训练信息和结果
├── output/                # 输出目录
│   ├── checkpoint-*/      # 训练检查点
│   └── confusion_matrix.png # 混淆矩阵
└── README.md              # 项目说明文档
```

## 环境配置

### 1. 创建Conda虚拟环境（推荐）

```bash
conda create -n distilbert_nlp python=3.9 -y
conda activate distilbert_nlp
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. GPU支持（可选但推荐）

如果使用GPU，需要安装CUDA版本的PyTorch：

```bash
# 卸载CPU版本
pip uninstall torch torchvision torchaudio

# 安装CUDA 12.1版本（兼容CUDA 12.9）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

验证GPU：
```bash
python -c "import torch; print('CUDA可用:', torch.cuda.is_available())"
```

### 4. 主要依赖包

- torch >= 2.0.0 (推荐CUDA版本)
- transformers >= 4.30.0
- datasets >= 2.12.0
- peft >= 0.4.0 (用于LoRA)
- accelerate >= 0.20.0
- scikit-learn >= 1.3.0
- shap >= 0.42.0 (用于SHAP分析)
- lime >= 0.2.0 (用于LIME分析)

## 使用方法

### 1. 准备数据

将IMDB数据集CSV文件放在 `./IMDB/` 目录下：
- `Train.csv` - 训练集（格式：text,label）
- `Test.csv` - 测试集
- `Valid.csv` - 验证集（可选）

### 2. 配置参数

编辑 `config.py` 文件，可以修改以下参数：

- **模型配置**: 模型名称、本地权重路径
- **数据集配置**: 数据集路径、批次大小、最大长度等
- **LoRA配置**: rank、alpha、dropout等（默认rank=16）
- **训练配置**: 学习率、训练轮数、保存步数等

### 3. 运行完整流程

```bash
# 运行所有步骤（训练、评估、可解释性分析）
python main.py --mode all
```

### 4. 分步运行

#### 训练模型

```bash
# 默认配置（使用config.py中的数据增强设置）
python main.py --mode train

# 启用数据增强
python main.py --mode train --use_data_augmentation true

# 禁用数据增强
python main.py --mode train --use_data_augmentation false

# 或直接运行
python train.py
```

训练时会显示：
- 训练配置信息
- GPU设备信息（如果可用）
- 预估训练时间
- 实时训练进度
- 实际训练时间和速度

#### 评估模型

```bash
python main.py --mode evaluate
# 或直接运行（自动使用最佳模型）
python evaluate.py
```

#### 可解释性分析

```bash
python main.py --mode explain
# 或直接运行
python explainability.py
```

## 数据集

### IMDb电影评论数据集

本项目使用本地IMDb数据集进行情感分析（二分类：正面/负面）。

**数据格式**: CSV文件，包含 `text` 和 `label` 两列
- `text`: 电影评论文本
- `label`: 标签（0=负面，1=正面）

### 使用其他数据集

如需使用其他数据集，需要：

1. 在 `data_loader.py` 中实现对应的数据加载函数
2. 修改 `config.py` 中的 `DATASET_NAME` 和 `DATASET_PATH` 参数

## 输出结果

### 训练输出

- **最佳模型**: `./saved_models/best_model/`（验证集F1分数最高的模型）
- **最终模型**: `./saved_models/final_model/`（训练完成时的模型）
- **训练信息**: `./saved_models/training_info.json`（包含训练配置和测试结果）
- **训练日志**: `./logs/`（可用TensorBoard查看：`tensorboard --logdir ./logs`）
- **检查点**: `./output/checkpoint-*/`（训练过程中的检查点）

### 评估输出

- 评估指标（准确率、F1分数等）打印在控制台
- 混淆矩阵图片保存在 `./output/confusion_matrix.png`
- 详细的分类报告打印在控制台

### 可解释性分析输出

所有可解释性分析结果保存在 `./output/explainability/`：

- `attention_visualization.png`: 注意力权重热力图
- `attention_visualization_bar.png`: Token重要性条形图
- `shap_analysis.png`: SHAP分析结果
- `lime_explanation_*.html`: LIME解释结果（HTML格式）

## 模型可解释性分析说明

### 1. 注意力可视化

- 展示模型在做出分类决策时，对输入文本中不同位置的关注程度
- 通过热力图和条形图直观展示重要词汇

### 2. SHAP分析

- 使用SHAP值量化每个token对最终预测的贡献
- 可以识别对分类结果影响最大的词语

### 3. LIME分析

- 通过局部可解释性模型解释单个样本的预测结果
- 生成HTML格式的交互式解释报告

## 性能优化

### GPU训练优化

项目已针对GPU训练进行优化：

- **批次大小**: 64（充分利用GPU并行能力）
- **序列长度**: 512（更好的文本理解）
- **LoRA rank**: 16（提升模型容量）
- **混合精度**: 自动启用FP16加速
- **数据加载**: 启用pin_memory加速数据传输

### 训练速度

- **GPU训练**: 约36样本/秒，3轮训练约55分钟（40,000样本）
- **CPU训练**: 约4-5样本/秒，预计需要数小时

### 显存使用

- RTX 4050 (6GB): 当前配置完全足够
- 如果显存不足，可以减小 `BATCH_SIZE` 或 `MAX_LENGTH`

## 配置说明

### 关键配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `BATCH_SIZE` | 64 | 批次大小（GPU推荐64，CPU推荐16-32） |
| `MAX_LENGTH` | 512 | 最大序列长度（GPU推荐512，CPU推荐256） |
| `LORA_R` | 32 | LoRA rank（GPU推荐32，CPU推荐4-8） |
| `LORA_ALPHA` | 64 | LoRA alpha（通常是rank的2倍） |
| `TARGET_MODULES` | `["q_lin", "v_lin", "k_lin", "out_lin", "fc1", "fc2"]` | LoRA目标模块（扩展版） |
| `NUM_EPOCHS` | 3 | 训练轮数 |
| `LEARNING_RATE` | 2e-5 | 学习率 |
| `USE_DATA_AUGMENTATION` | `True` | 是否使用数据增强 |
| `DATA_AUGMENTATION_RATE` | `0.3` | 数据增强比例（0-1） |

## 常见问题

### Q: 如何修改分类任务的类别数？

A: 修改 `config.py` 中的相关配置，并在 `data_loader.py` 中确保标签数量正确。

### Q: 如何使用自己的数据集？

A: 
1. 准备CSV文件（格式：text,label）
2. 将文件放在 `./IMDB/` 目录（或修改 `DATASET_PATH`）
3. 修改 `config.py` 中的 `TRAIN_FILE`、`TEST_FILE` 等参数

### Q: LoRA参数如何选择？

A: 
- GPU环境：`LORA_R=16`, `LORA_ALPHA=32`（当前配置）
- CPU环境：`LORA_R=4-8`, `LORA_ALPHA=8-16`
- 可以尝试不同组合找到最佳效果

### Q: 如何检查GPU是否可用？

A: 
```bash
python -c "import torch; print('CUDA可用:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

### Q: 训练时间预估不准确？

A: 预估时间基于经验值，实际时间取决于GPU性能、批次大小等因素。训练开始后会显示实际速度。

### Q: 可解释性分析很慢怎么办？

A: 减少 `config.py` 中的 `NUM_SAMPLES_FOR_EXPLAIN` 参数，只分析少量样本。

### Q: 如何调整数据增强参数？

A: 在 `config.py` 中修改：
- `USE_DATA_AUGMENTATION`: 是否使用数据增强
- `DATA_AUGMENTATION_RATE`: 数据增强比例（0-1）

### Q: 如何使用模型集成功能？

A: 参考 `ensemble.py` 中的函数：
- `load_ensemble_models`: 加载多个模型用于集成
- `ensemble_predict`: 平均概率集成预测
- `weighted_ensemble_predict`: 加权平均集成预测
- `vote_ensemble_predict`: 投票集成预测

## 训练结果示例

使用当前配置在IMDb数据集上的训练结果：

- **训练时间**: 约55分钟（GPU）
- **训练速度**: 36.04 样本/秒
- **测试集准确率**: 94.50%
- **测试集F1分数**: 94.50%
- **可训练参数**: 1,774,082 (仅占2.62%)

## 参考文献

- DistilBERT: [Sanh et al., 2019](https://arxiv.org/abs/1910.01108)
- LoRA: [Hu et al., 2021](https://arxiv.org/abs/2106.09685)
- SHAP: [Lundberg & Lee, 2017](https://arxiv.org/abs/1705.07874)
- LIME: [Ribeiro et al., 2016](https://arxiv.org/abs/1602.04938)

## 许可证

本项目仅供学习和研究使用。

## 更新日志

- **v1.0**: 初始版本，支持LoRA微调和可解释性分析
- **v1.1**: 添加GPU训练支持，优化训练参数，提升准确率至90.10%
- **v1.2**: 添加本地数据集支持，优化训练时间显示，自动保存最佳模型
