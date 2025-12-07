# -*- coding: utf-8 -*-
"""
模型评估脚本：在测试集上评估模型性能

主要功能：
- evaluate_model(): 评估模型在测试集上的性能，计算准确率、F1分数等指标

调用关系：
- 调用 data_loader.prepare_datasets() (data_loader.py第129行) 准备测试数据集
- 调用 model_utils.get_model_for_inference() (model_utils.py第172行) 加载推理模型
"""
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import Config  # 配置文件，定义评估相关参数
from data_loader import prepare_datasets, TextClassificationDataset  # 数据加载模块，定义在data_loader.py
from model_utils import get_model_for_inference  # 模型工具，定义在model_utils.py第172行


def evaluate_model(model_path, dataset_name="imdb"):
    """
    评估模型性能
    
    调用流程：
    1. 调用 prepare_datasets() (data_loader.py第129行) 准备测试数据集
    2. 调用 get_model_for_inference() (model_utils.py第172行) 加载推理模型
    3. 使用DataLoader批量处理数据
    4. 计算评估指标（准确率、F1分数等）
    5. 绘制混淆矩阵并保存
    
    Args:
        model_path: 模型路径（LoRA权重目录）
        dataset_name: 数据集名称
        
    Returns:
        dict: 包含准确率、F1分数、精确率、召回率等指标的字典
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 准备数据集
    # 调用data_loader.py第129行的prepare_datasets函数
    tokenizer, _, _, test_dataset = prepare_datasets(dataset_name)
    
    # 加载模型
    num_labels = len(set([item['labels'].item() for item in test_dataset]))
    # 调用model_utils.py第172行的get_model_for_inference函数
    model = get_model_for_inference(model_path, num_labels=num_labels)
    model.to(device)
    
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False
    )
    
    # 评估
    model.eval()
    all_predictions = []
    all_labels = []
    
    print("正在评估...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    
    print("\n" + "="*50)
    print("评估结果:")
    print("="*50)
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"F1分数: {f1:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print("="*50)
    
    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(all_labels, all_predictions))
    
    # 绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'confusion_matrix.png'))
    print(f"\n混淆矩阵已保存到: {os.path.join(Config.OUTPUT_DIR, 'confusion_matrix.png')}")
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'predictions': all_predictions,
        'labels': all_labels
    }


if __name__ == "__main__":
    # 优先使用最佳模型，如果不存在则使用最终模型
    best_model_path = os.path.join(Config.SAVE_MODEL_DIR, "best_model")
    final_model_path = os.path.join(Config.SAVE_MODEL_DIR, "final_model")
    
    if os.path.exists(best_model_path):
        print("使用最佳模型进行评估...")
        model_path = best_model_path
    elif os.path.exists(final_model_path):
        print("使用最终模型进行评估...")
        model_path = final_model_path
    else:
        print(f"模型路径不存在: {best_model_path} 或 {final_model_path}")
        print("请先运行 train.py 训练模型")
        exit(1)
    
    evaluate_model(model_path, Config.DATASET_NAME)



