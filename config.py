# -*- coding: utf-8 -*-
"""
配置文件：定义模型和训练的超参数
"""
import os


class Config:
    """项目配置类"""

    # 模型配置
    MODEL_NAME = "distilbert-base-uncased"  # 基础模型名称
    MODEL_PATH = "./pytorch_model.bin"  # 本地模型权重路径

    # 数据集配置
    DATASET_NAME = "imdb"  # 可选: "imdb", "thucnews"等
    DATASET_PATH = "./IMDB"  # 本地数据集路径
    TRAIN_FILE = "Train.csv"  # 训练集文件名
    TEST_FILE = "Test.csv"  # 测试集文件名
    VALID_FILE = "Valid.csv"  # 验证集文件名（如果存在）
    USE_LOCAL_DATASET = True  # 是否使用本地数据集
    MAX_LENGTH = 512  # 最大序列长度（GPU可用，可以增加到512）
    BATCH_SIZE = 64  # 批次大小（GPU可用，增加到64）
    GRADIENT_ACCUMULATION_STEPS = 1  # 梯度累积步数（GPU显存足够，可以设为1）
    VALIDATION_SPLIT = 0.1  # 验证集比例（如果本地没有验证集，从训练集划分）
    NUM_WORKERS = 4  # 数据加载的进程数（根据CPU核心数调整）

    # LoRA配置
    USE_LORA = True  # 是否使用LoRA微调
    #LORA_R = 16  # LoRA的rank（GPU可用，增加到16提升性能）
    #LORA_ALPHA = 32  # LoRA的alpha参数（通常是rank的2倍）
    LORA_DROPOUT = 0.1  # LoRA的dropout率
    # LoRA目标模块：对于DistilBERT，通常是["q_lin", "v_lin"]
    # 如果遇到错误，可以尝试["query", "value"]或使用auto模式
    #TARGET_MODULES = ["q_lin", "v_lin"]  # LoRA目标模块
    LORA_R = 32
    LORA_ALPHA = 64
    TARGET_MODULES = ["q_lin", "v_lin", "k_lin", "out_lin", "fc1", "fc2"]

    # 训练配置
    NUM_EPOCHS = 3  # 训练轮数（GPU可用，增加到3轮提升性能）
    LEARNING_RATE = 2e-5  # 学习率
    WEIGHT_DECAY = 0.01  # 权重衰减
    WARMUP_STEPS = 500  # 预热步数（GPU可用，可以增加）
    LOGGING_STEPS = 100  # 日志记录步数
    SAVE_STEPS = 500  # 模型保存步数（GPU训练更快，可以更频繁保存）
    EVAL_STEPS = 500  # 评估步数（GPU评估更快，可以更频繁评估）
    MAX_TRAIN_SAMPLES = None  # 最大训练样本数（None表示使用全部）

    # 输出配置
    OUTPUT_DIR = "./output"  # 输出目录（训练过程中的检查点）
    LOG_DIR = "./logs"  # 日志目录
    SAVE_MODEL_DIR = "./saved_models"  # 保存模型目录（最终模型和最佳模型）
    AUTO_SAVE = True  # 是否自动保存模型（训练完成后）

    # 可解释性配置
    NUM_SAMPLES_FOR_EXPLAIN = 10  # 用于可解释性分析的样本数量
    USE_ATTENTION_VIZ = True  # 是否使用注意力可视化
    USE_SHAP = True  # 是否使用SHAP
    USE_LIME = True  # 是否使用LIME

    # 数据增强配置
    USE_DATA_AUGMENTATION = True  # 是否使用数据增强
    DATA_AUGMENTATION_RATE = 0.3  # 数据增强比例（0-1）

    @classmethod
    def create_dirs(cls):
        """创建必要的目录"""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs(cls.SAVE_MODEL_DIR, exist_ok=True)