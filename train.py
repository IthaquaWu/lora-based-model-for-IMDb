# -*- coding: utf-8 -*-
"""
模型训练脚本：使用LoRA进行参数高效微调

主要功能：
- train(): 主训练函数，整合数据加载、模型训练、评估和保存流程

调用关系：
- 调用 data_loader.prepare_datasets() (data_loader.py第129行) 准备数据集
- 调用 model_utils.get_model_for_training() (model_utils.py第157行) 获取训练模型
- 调用 compute_metrics() (本文件第16行) 计算评估指标
"""
import os
import torch
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from config import Config  # 配置文件，定义训练超参数
from data_loader import prepare_datasets  # 数据加载模块，定义在data_loader.py第129行
from model_utils import get_model_for_training  # 模型工具，定义在model_utils.py第157行


def compute_metrics(eval_pred):
    """
    计算评估指标
    
    Args:
        eval_pred: 评估预测结果
        
    Returns:
        dict: 包含准确率、F1分数等的字典
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def train():
    """
    主训练函数
    
    调用流程：
    1. 调用 Config.create_dirs() (config.py第57行) 创建输出目录
    2. 调用 prepare_datasets() (data_loader.py第129行) 准备数据集
    3. 调用 get_model_for_training() (model_utils.py第157行) 获取训练模型
    4. 调用 compute_metrics() (本文件第16行) 计算评估指标
    5. 调用 trainer.train() (transformers库) 执行训练
    6. 调用 trainer.evaluate() (transformers库) 在测试集上评估
    
    Returns:
        trainer: Trainer对象
        test_results: 测试集评估结果
    """
    # 创建输出目录
    Config.create_dirs()  # 调用config.py第57行的create_dirs方法
    
    # 准备数据集
    # 调用data_loader.py第129行的prepare_datasets函数
    tokenizer, train_dataset, val_dataset, test_dataset = prepare_datasets(
        Config.DATASET_NAME,
        data_augmentation=Config.USE_DATA_AUGMENTATION,
        augment_rate=Config.DATA_AUGMENTATION_RATE
    )
    
    # 加载模型
    # 安全地获取标签数量，处理Subset对象情况
    try:
        # 尝试获取前100个样本的标签来确定类别数（提高性能）
        sample_size = min(100, len(train_dataset))
        labels = []
        for i in range(sample_size):
            try:
                item = train_dataset[i]
                label = item['labels']
                if hasattr(label, 'item'):
                    label = label.item()
                labels.append(label)
            except Exception as e:
                continue
        num_labels = len(set(labels))
        # 如果没有获取到足够的标签，使用默认值
        if num_labels == 0:
            num_labels = 2  # 默认二分类
    except Exception as e:
        print(f"获取标签数量时出错: {e}")
        num_labels = 2  # 默认二分类
    
    print(f"检测到的类别数: {num_labels}")
    # 调用model_utils.py第157行的get_model_for_training函数
    model = get_model_for_training(num_labels=num_labels)
    
    # 数据整理器
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        num_train_epochs=Config.NUM_EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE * 2,  # 评估时使用更大的批次
        gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,  # 梯度累积
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        warmup_steps=Config.WARMUP_STEPS,
        logging_dir=Config.LOG_DIR,
        logging_steps=Config.LOGGING_STEPS,
        save_steps=Config.SAVE_STEPS,
        eval_steps=Config.EVAL_STEPS,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,  # 只保留最近3个检查点，节省空间
        load_best_model_at_end=True,  # 训练结束后加载最佳模型
        metric_for_best_model="f1",
        greater_is_better=True,
        save_on_each_node=False,  # 单机训练设为False
        report_to="tensorboard",
        fp16=torch.cuda.is_available(),  # 如果使用GPU，启用混合精度训练
        bf16=False,  # 如果GPU支持bfloat16，可以启用（RTX 4050不支持）
        dataloader_num_workers=0,  # 数据加载器工作进程数（Windows上设为0避免多进程问题）
        dataloader_pin_memory=torch.cuda.is_available(),  # GPU可用时启用pin_memory加速数据传输
        remove_unused_columns=True,  # 移除未使用的列
        optim="adamw_torch",  # 使用AdamW优化器
        max_grad_norm=1.0,  # 梯度裁剪
    )
    
    # 如果设置了最大训练样本数，限制训练集大小
    if Config.MAX_TRAIN_SAMPLES and Config.MAX_TRAIN_SAMPLES > 0:
        print(f"限制训练样本数为: {Config.MAX_TRAIN_SAMPLES}")
        import random
        indices = list(range(len(train_dataset)))
        random.seed(42)
        random.shuffle(indices)
        selected_indices = indices[:Config.MAX_TRAIN_SAMPLES]
        train_dataset = torch.utils.data.Subset(train_dataset, selected_indices)
        print(f"实际训练样本数: {len(train_dataset)}")
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,  # 使用本文件第16行定义的compute_metrics函数
    )
    
    # 开始训练
    print("\n" + "="*60)
    print("开始训练...")
    print("="*60)
    
    # 计算训练步数和预估时间
    total_steps = len(train_dataset) // (Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION_STEPS) * Config.NUM_EPOCHS
    print(f"\n训练配置:")
    print(f"  训练样本数: {len(train_dataset)}")
    print(f"  批次大小: {Config.BATCH_SIZE}")
    print(f"  梯度累积步数: {Config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"  有效批次大小: {Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"  训练轮数: {Config.NUM_EPOCHS}")
    print(f"  最大序列长度: {Config.MAX_LENGTH}")
    print(f"  LoRA rank: {Config.LORA_R}")
    print(f"  总训练步数: 约 {total_steps} 步")
    
    # 预估训练时间（基于GPU速度）
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"\nGPU设备: {device_name}")
        # GPU训练速度估算（根据批次大小和序列长度调整）
        # RTX 4050 + batch_size=64 + max_length=512，预计每秒15-25步
        estimated_steps_per_second = 20  # GPU训练速度估算
        estimated_seconds = total_steps / estimated_steps_per_second
        estimated_hours = int(estimated_seconds // 3600)
        estimated_minutes = int((estimated_seconds % 3600) // 60)
        print(f"\n预估训练时间（GPU）:")
        print(f"  预估速度: {estimated_steps_per_second} 步/秒")
        if estimated_hours >= 1:
            print(f"  预估时间: 约 {estimated_hours} 小时 {estimated_minutes} 分钟")
        else:
            print(f"  预估时间: 约 {estimated_minutes} 分钟")
    else:
        estimated_steps_per_second = 0.1  # CPU训练速度估算
        estimated_seconds = total_steps / estimated_steps_per_second
        estimated_hours = int(estimated_seconds // 3600)
        print(f"\n预估训练时间（CPU）:")
        print(f"  预估速度: {estimated_steps_per_second} 步/秒")
        print(f"  预估时间: 约 {estimated_hours} 小时")
    
    print("="*60 + "\n")
    
    import time
    start_time = time.time()
    train_result = trainer.train()
    end_time = time.time()
    
    # 显示实际训练时间
    actual_time = end_time - start_time
    actual_hours = int(actual_time // 3600)
    actual_minutes = int((actual_time % 3600) // 60)
    actual_seconds = int(actual_time % 60)
    
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print(f"实际训练时间: {actual_hours} 小时 {actual_minutes} 分钟 {actual_seconds} 秒")
    print(f"总训练步数: {train_result.global_step}")
    if hasattr(train_result, 'metrics') and 'train_samples_per_second' in train_result.metrics:
        print(f"实际训练速度: {train_result.metrics['train_samples_per_second']:.2f} 样本/秒")
        print(f"实际步速: {train_result.global_step / actual_time:.2f} 步/秒")
    print("="*60 + "\n")
    
    # 保存最终模型（训练完成后的模型）
    final_model_path = os.path.join(Config.SAVE_MODEL_DIR, "final_model")
    print(f"\n正在保存最终模型到: {final_model_path}")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"✅ 最终模型已保存到: {final_model_path}")
    
    # 保存最佳模型（如果启用了load_best_model_at_end，最佳模型已经在trainer中）
    # 由于设置了load_best_model_at_end=True，trainer.model已经是验证集上表现最好的模型
    best_model_path = os.path.join(Config.SAVE_MODEL_DIR, "best_model")
    print(f"\n正在保存最佳模型到: {best_model_path}")
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    print(f"✅ 最佳模型已保存到: {best_model_path}")
    
    # 保存训练指标
    import json
    training_info = {
        "train_loss": train_result.training_loss if hasattr(train_result, 'training_loss') else None,
        "train_runtime": train_result.metrics.get('train_runtime', None) if hasattr(train_result, 'metrics') else None,
        "train_samples_per_second": train_result.metrics.get('train_samples_per_second', None) if hasattr(train_result, 'metrics') else None,
        "epoch": train_result.metrics.get('epoch', None) if hasattr(train_result, 'metrics') else None,
        "config": {
            "num_epochs": Config.NUM_EPOCHS,
            "batch_size": Config.BATCH_SIZE,
            "learning_rate": Config.LEARNING_RATE,
            "max_length": Config.MAX_LENGTH,
            "lora_r": Config.LORA_R,
            "lora_alpha": Config.LORA_ALPHA,
        }
    }
    
    # 在测试集上评估
    print("\n在测试集上评估...")
    test_results = trainer.evaluate(test_dataset)
    print("测试集结果:")
    for key, value in test_results.items():
        print(f"  {key}: {value:.4f}")
    
    # 将测试结果添加到训练信息中
    training_info["test_results"] = test_results
    
    # 保存训练信息到JSON文件
    info_path = os.path.join(Config.SAVE_MODEL_DIR, "training_info.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(training_info, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 训练信息已保存到: {info_path}")
    
    # 打印保存位置总结
    print("\n" + "="*60)
    print("模型保存完成！")
    print("="*60)
    print(f"最终模型: {final_model_path}")
    print(f"最佳模型: {best_model_path}")
    print(f"训练信息: {info_path}")
    print(f"检查点目录: {Config.OUTPUT_DIR}")
    print("="*60)
    
    return trainer, test_results


if __name__ == "__main__":
    train()


