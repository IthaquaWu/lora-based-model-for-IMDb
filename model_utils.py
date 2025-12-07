# -*- coding: utf-8 -*-
"""
模型工具函数：加载模型、设置LoRA等

本模块提供模型加载和LoRA配置功能：
- load_model(): 加载预训练模型（优先从本地加载）
- setup_lora(): 配置LoRA参数高效微调
- get_model_for_training(): 获取用于训练的模型（包含LoRA）
- get_model_for_inference(): 获取用于推理的模型（加载LoRA权重）
"""
import os
from transformers import AutoModelForSequenceClassification, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType
import torch
from config import Config  # 配置文件，定义模型路径、LoRA参数等


def load_model(num_labels=2):
    """
    加载预训练模型（优先从本地加载）
    
    Args:
        num_labels: 分类标签数量
        
    Returns:
        model: 加载的模型
    """
    print(f"正在加载模型: {Config.MODEL_NAME}")
    
    # 优先从本地加载模型权重
    if Config.MODEL_PATH and os.path.exists(Config.MODEL_PATH):
        print(f"从本地加载模型权重: {Config.MODEL_PATH}")
        try:
            # 尝试从本地加载配置
            try:
                config = AutoConfig.from_pretrained(
                    Config.MODEL_NAME,
                    num_labels=num_labels,
                    local_files_only=True
                )
            except:
                # 如果本地没有配置，从Hugging Face加载
                print("本地未找到模型配置，从Hugging Face加载配置...")
                config = AutoConfig.from_pretrained(
                    Config.MODEL_NAME,
                    num_labels=num_labels
                )
            
            # 创建模型结构
            model = AutoModelForSequenceClassification.from_pretrained(
                Config.MODEL_NAME,
                config=config,
                ignore_mismatched_sizes=True,
                local_files_only=False  # 需要模型结构，可能需要从HF加载
            )
            
            # 加载本地权重
            print(f"加载本地权重文件: {Config.MODEL_PATH}")
            state_dict = torch.load(Config.MODEL_PATH, map_location='cpu')
            
            # 尝试加载权重
            try:
                model.load_state_dict(state_dict, strict=False)
                print("成功加载本地模型权重")
            except Exception as e:
                print(f"加载权重时出现警告（可能因为模型结构不完全匹配）: {e}")
                # 尝试部分加载
                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
                print("部分加载权重成功")
            
            return model
        except Exception as e:
            print(f"从本地加载模型失败: {e}")
            print("尝试从Hugging Face加载模型...")
    
    # 如果本地加载失败，从Hugging Face加载
    print("从Hugging Face加载模型...")
    try:
        config = AutoConfig.from_pretrained(
            Config.MODEL_NAME,
            num_labels=num_labels
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            Config.MODEL_NAME,
            config=config,
            ignore_mismatched_sizes=True
        )
        
        # 如果存在本地权重文件，尝试加载权重
        if Config.MODEL_PATH and os.path.exists(Config.MODEL_PATH):
            print(f"尝试加载本地权重: {Config.MODEL_PATH}")
            try:
                state_dict = torch.load(Config.MODEL_PATH, map_location='cpu')
                model.load_state_dict(state_dict, strict=False)
                print("成功加载本地权重")
            except Exception as e:
                print(f"加载本地权重失败: {e}")
    except Exception as e:
        print(f"加载模型时出错: {e}")
        raise
    
    return model


def setup_lora(model):
    """
    设置LoRA参数高效微调
    
    Args:
        model: 基础模型
        
    Returns:
        model: 配置了LoRA的模型
    """
    if not Config.USE_LORA:
        print("未启用LoRA，使用全参数微调")
        return model
    
    print("正在配置LoRA...")
    
    # 定义LoRA配置
    try:
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,  # 序列分类任务
            r=Config.LORA_R,  # LoRA的rank
            lora_alpha=Config.LORA_ALPHA,  # LoRA的alpha参数
            lora_dropout=Config.LORA_DROPOUT,  # LoRA的dropout率
            target_modules=Config.TARGET_MODULES,  # 目标模块
            bias="none",  # 不训练bias
        )
        
        # 应用LoRA
        model = get_peft_model(model, lora_config)
    except Exception as e:
        print(f"使用指定目标模块配置LoRA失败: {e}")
        print("尝试使用自动检测模式...")
        # 如果指定的模块名称不匹配，尝试自动检测
        try:
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=Config.LORA_R,
                lora_alpha=Config.LORA_ALPHA,
                lora_dropout=Config.LORA_DROPOUT,
                target_modules="all-linear",  # 尝试所有线性层
                bias="none",
            )
            model = get_peft_model(model, lora_config)
            print("成功使用自动检测模式配置LoRA")
        except Exception as e2:
            print(f"自动检测模式也失败: {e2}")
            print("请检查模型结构或手动指定正确的target_modules")
            raise
    
    # 打印可训练参数信息
    model.print_trainable_parameters()
    
    return model


def get_model_for_training(num_labels=2):
    """
    获取用于训练的模型（包含LoRA配置）
    
    调用流程：
    1. 调用 load_model() (本文件第12行) 加载基础模型
    2. 调用 setup_lora() (本文件第102行) 配置LoRA适配器
    
    Args:
        num_labels: 分类标签数量
        
    Returns:
        model: 配置好LoRA的模型，可直接用于训练
    """
    model = load_model(num_labels)  # 调用本文件第12行的load_model函数
    model = setup_lora(model)  # 调用本文件第102行的setup_lora函数
    return model


def get_model_for_inference(model_path, num_labels=2):
    """
    获取用于推理的模型（优先从本地加载）
    
    Args:
        model_path: 模型路径
        num_labels: 分类标签数量
        
    Returns:
        model: 加载的模型
    """
    from peft import PeftModel
    
    # 首先尝试加载基础模型（优先使用本地权重）
    if Config.MODEL_PATH and os.path.exists(Config.MODEL_PATH):
        print("使用本地基础模型权重")
        base_model = load_model(num_labels=num_labels)  # 调用本文件第12行的load_model函数
    else:
        # 从Hugging Face加载基础模型
        base_model = AutoModelForSequenceClassification.from_pretrained(
            Config.MODEL_NAME,
            num_labels=num_labels
        )
    
    # 如果存在LoRA权重，加载LoRA权重
    if os.path.exists(model_path):
        try:
            model = PeftModel.from_pretrained(base_model, model_path)
            print(f"成功加载LoRA模型: {model_path}")
        except:
            # 如果没有LoRA权重，尝试直接加载完整模型
            try:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_path,
                    num_labels=num_labels
                )
                print(f"成功加载完整模型: {model_path}")
            except:
                # 如果都失败，使用基础模型
                print(f"无法加载 {model_path}，使用基础模型")
                model = base_model
    else:
        print(f"模型路径不存在: {model_path}，使用基础模型")
        model = base_model
    
    model.eval()
    return model

