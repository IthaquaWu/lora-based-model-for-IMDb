"""模型集成模块"""
import torch
import numpy as np
from typing import List, Tuple, Any
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from config import Config


def load_ensemble_models(model_paths: List[str], device: torch.device) -> List[Any]:
    """
    加载多个模型用于集成
    
    Args:
        model_paths: 模型路径列表
        device: 设备（CPU/GPU）
        
    Returns:
        加载好的模型列表
    """
    models = []
    
    for path in model_paths:
        print(f"正在加载模型: {path}")
        try:
            # 加载模型
            model = AutoModelForSequenceClassification.from_pretrained(
                Config.MODEL_NAME, 
                local_files_only=True,
                num_labels=2  # 分类数，根据数据集调整
            )
            
            # 加载权重
            model.load_state_dict(torch.load(path, map_location=device))
            model.to(device)
            model.eval()
            models.append(model)
            print(f"模型 {path} 加载成功")
        except Exception as e:
            print(f"加载模型 {path} 失败: {e}")
    
    return models


def ensemble_predict(models: List[Any], tokenizer: AutoTokenizer, text: str, device: torch.device) -> Tuple[int, np.ndarray]:
    """
    使用集成模型进行预测
    
    Args:
        models: 模型列表
        tokenizer: 分词器
        text: 输入文本
        device: 设备（CPU/GPU）
        
    Returns:
        预测标签和平均概率分布
    """
    predictions = []
    
    for model in models:
        model.eval()
        with torch.no_grad():
            encoding = tokenizer(
                text, 
                truncation=True,
                padding='max_length',
                max_length=Config.MAX_LENGTH,
                return_tensors='pt'
            ).to(device)
            outputs = model(**encoding)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
            predictions.append(probs)
    
    # 平均概率
    avg_probs = np.mean(predictions, axis=0)
    
    return np.argmax(avg_probs), avg_probs


def ensemble_evaluate(models: List[Any], tokenizer: AutoTokenizer, dataset: Any, device: torch.device) -> float:
    """
    评估集成模型的性能
    
    Args:
        models: 模型列表
        tokenizer: 分词器
        dataset: 数据集
        device: 设备（CPU/GPU）
        
    Returns:
        集成模型的准确率
    """
    correct = 0
    total = 0
    
    for batch in dataset:
        inputs = {k: v.to(device) for k, v in batch.items()}
        text = batch['text']  # 假设dataset返回文本
        labels = inputs['labels'].cpu().numpy()
        
        predictions = []
        for model in models:
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                predictions.append(probs)
        
        # 平均概率
        avg_probs = np.mean(predictions, axis=0)
        predicted_labels = np.argmax(avg_probs, axis=1)
        
        # 计算准确率
        correct += np.sum(predicted_labels == labels)
        total += len(labels)
    
    return correct / total


def weighted_ensemble_predict(models: List[Any], tokenizer: AutoTokenizer, text: str, device: torch.device, weights: List[float] = None) -> Tuple[int, np.ndarray]:
    """
    加权集成预测
    
    Args:
        models: 模型列表
        tokenizer: 分词器
        text: 输入文本
        device: 设备（CPU/GPU）
        weights: 模型权重列表
        
    Returns:
        预测标签和加权平均概率分布
    """
    if weights is None:
        weights = [1.0] * len(models)
    
    # 标准化权重
    weights = np.array(weights) / np.sum(weights)
    
    predictions = []
    
    for model in models:
        model.eval()
        with torch.no_grad():
            encoding = tokenizer(
                text, 
                truncation=True,
                padding='max_length',
                max_length=Config.MAX_LENGTH,
                return_tensors='pt'
            ).to(device)
            outputs = model(**encoding)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
            predictions.append(probs)
    
    # 加权平均概率
    weighted_probs = np.average(predictions, axis=0, weights=weights)
    
    return np.argmax(weighted_probs), weighted_probs


def vote_ensemble_predict(models: List[Any], tokenizer: AutoTokenizer, text: str, device: torch.device) -> int:
    """
    投票集成预测
    
    Args:
        models: 模型列表
        tokenizer: 分词器
        text: 输入文本
        device: 设备（CPU/GPU）
        
    Returns:
        投票结果
    """
    votes = []
    
    for model in models:
        model.eval()
        with torch.no_grad():
            encoding = tokenizer(
                text, 
                truncation=True,
                padding='max_length',
                max_length=Config.MAX_LENGTH,
                return_tensors='pt'
            ).to(device)
            outputs = model(**encoding)
            predicted_label = torch.argmax(outputs.logits, dim=-1).cpu().item()
            votes.append(predicted_label)
    
    # 统计投票
    return max(set(votes), key=votes.count)
