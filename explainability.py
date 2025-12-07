# -*- coding: utf-8 -*-
"""
模型可解释性分析：注意力可视化、SHAP和LIME分析

主要功能：
- visualize_attention(): 可视化注意力权重
- shap_analysis(): 使用SHAP进行可解释性分析
- lime_analysis(): 使用LIME进行可解释性分析
- analyze_explainability(): 完整的可解释性分析流程

调用关系：
- 调用 data_loader.prepare_datasets() (data_loader.py第129行) 准备数据集
- 调用 model_utils.get_model_for_inference() (model_utils.py第172行) 加载推理模型
- 调用 visualize_attention() (本文件第63行) 进行注意力可视化
- 调用 shap_analysis() (本文件第151行) 进行SHAP分析
- 调用 lime_analysis() (本文件第185行) 进行LIME分析
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import shap
from lime.lime_text import LimeTextExplainer
from config import Config  # 配置文件，定义可解释性分析参数
from data_loader import prepare_datasets  # 数据加载模块，定义在data_loader.py第129行
from model_utils import get_model_for_inference  # 模型工具，定义在model_utils.py第172行


class ModelWrapper:
    """模型包装器，用于SHAP和LIME"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def __call__(self, texts):
        """
        对文本列表进行预测
        
        Args:
            texts: 文本列表
            
        Returns:
            predictions: 预测概率
        """
        if isinstance(texts, str):
            texts = [texts]
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for text in texts:
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=Config.MAX_LENGTH,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1)
                predictions.append(probs.cpu().numpy()[0])
        
        return np.array(predictions)


def visualize_attention(model, tokenizer, text, device, save_path=None):
    """
    可视化注意力权重
    
    Args:
        model: 模型
        tokenizer: 分词器
        text: 输入文本
        device: 设备
        save_path: 保存路径
    """
    model.eval()
    
    # 编码文本
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=Config.MAX_LENGTH,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # 获取注意力权重
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        attentions = outputs.attentions  # 所有层的注意力
    
    # 使用最后一层的平均注意力
    attention = attentions[-1].mean(dim=1).squeeze().cpu().numpy()
    
    # 获取tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # 可视化
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 只显示非padding的tokens
    valid_length = attention_mask[0].sum().item()
    attention = attention[:valid_length, :valid_length]
    tokens = tokens[:valid_length]
    
    # 计算每个token的平均注意力（来自所有其他tokens）
    token_attention = attention.mean(axis=0)
    
    # 绘制热力图
    im = ax.imshow(attention, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(tokens, fontsize=8)
    ax.set_title('注意力权重可视化')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"注意力可视化已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    # 绘制token重要性条形图
    fig, ax = plt.subplots(figsize=(12, 6))
    top_indices = np.argsort(token_attention)[-20:][::-1]  # 前20个最重要的tokens
    top_tokens = [tokens[i] for i in top_indices]
    top_attention = token_attention[top_indices]
    
    ax.barh(range(len(top_tokens)), top_attention)
    ax.set_yticks(range(len(top_tokens)))
    ax.set_yticklabels(top_tokens)
    ax.set_xlabel('平均注意力权重')
    ax.set_title('Top 20 最重要的Tokens')
    plt.tight_layout()
    
    if save_path:
        bar_path = save_path.replace('.png', '_bar.png')
        plt.savefig(bar_path, dpi=300, bbox_inches='tight')
        print(f"Token重要性条形图已保存到: {bar_path}")
    else:
        plt.show()
    
    plt.close()


def shap_analysis(model_wrapper, texts, class_names=None, save_path=None):
    """
    使用SHAP进行可解释性分析
    
    Args:
        model_wrapper: 模型包装器
        texts: 文本列表
        class_names: 类别名称
        save_path: 保存路径
    """
    print("正在进行SHAP分析...")
    
    # 创建SHAP解释器
    explainer = shap.Explainer(model_wrapper, model_wrapper.tokenizer)
    
    # 计算SHAP值（只分析前几个样本以节省时间）
    num_samples = min(len(texts), Config.NUM_SAMPLES_FOR_EXPLAIN)
    shap_values = explainer(texts[:num_samples])
    
    # 可视化
    if class_names is None:
        class_names = ['Negative', 'Positive']
    
    shap.plots.text(shap_values[:, :, 1], display=False)  # 显示正类的SHAP值
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"SHAP分析结果已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def lime_analysis(model_wrapper, texts, class_names=None, save_dir=None):
    """
    使用LIME进行可解释性分析
    
    Args:
        model_wrapper: 模型包装器
        texts: 文本列表
        class_names: 类别名称
        save_dir: 保存目录
    """
    print("正在进行LIME分析...")
    
    if class_names is None:
        class_names = ['Negative', 'Positive']
    
    explainer = LimeTextExplainer(class_names=class_names)
    
    num_samples = min(len(texts), Config.NUM_SAMPLES_FOR_EXPLAIN)
    
    for i, text in enumerate(texts[:num_samples]):
        print(f"\n分析样本 {i+1}/{num_samples}...")
        
        # 定义预测函数
        def predict_proba(texts):
            return model_wrapper(texts)
        
        # 解释
        explanation = explainer.explain_instance(
            text,
            predict_proba,
            num_features=20,
            top_labels=1
        )
        
        # 保存解释结果
        if save_dir:
            save_path = os.path.join(save_dir, f'lime_explanation_{i+1}.html')
            explanation.save_to_file(save_path)
            print(f"LIME解释结果已保存到: {save_path}")
        else:
            print(explanation.as_list())


def analyze_explainability(model_path, dataset_name="imdb"):
    """
    进行完整的可解释性分析
    
    调用流程：
    1. 调用 prepare_datasets() (data_loader.py第129行) 准备数据集
    2. 调用 get_model_for_inference() (model_utils.py第172行) 加载推理模型
    3. 调用 visualize_attention() (本文件第63行) 进行注意力可视化
    4. 调用 shap_analysis() (本文件第151行) 进行SHAP分析
    5. 调用 lime_analysis() (本文件第185行) 进行LIME分析
    
    Args:
        model_path: 模型路径（LoRA权重目录）
        dataset_name: 数据集名称
        
    输出：
        所有结果自动保存到 ./output/explainability/ 目录
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
    
    # 创建模型包装器
    model_wrapper = ModelWrapper(model, tokenizer, device)
    
    # 获取一些样本用于分析
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    # 收集样本文本和标签
    sample_texts = []
    sample_labels = []
    
    # 从数据集中获取原始文本
    for i, item in enumerate(test_dataset):
        if i >= Config.NUM_SAMPLES_FOR_EXPLAIN:
            break
        if 'text' in item:
            sample_texts.append(item['text'])
            sample_labels.append(item['labels'].item())
        else:
            # 如果没有保存原始文本，使用示例文本
            break
    
    # 如果没有收集到文本，使用示例文本
    if not sample_texts:
        sample_texts = [
            "This movie is terrible. I hated every minute of it. The acting was awful and the plot made no sense.",
            "This movie is absolutely fantastic! I loved every minute of it. The acting was superb and the plot was engaging.",
            "The film was okay, nothing special. Some parts were good, others were boring.",
        ]
        sample_labels = [0, 1, 0]
    
    # 创建输出目录
    explain_dir = os.path.join(Config.OUTPUT_DIR, "explainability")
    os.makedirs(explain_dir, exist_ok=True)
    
    # 注意力可视化
    if Config.USE_ATTENTION_VIZ:
        print("\n进行注意力可视化...")
        # 使用一个示例文本
        example_text = "This movie is absolutely fantastic! I loved every minute of it."
        # 调用本文件第63行定义的visualize_attention函数
        visualize_attention(
            model, tokenizer, example_text, device,
            save_path=os.path.join(explain_dir, "attention_visualization.png")
        )
    
    # SHAP分析
    if Config.USE_SHAP:
        print("\n进行SHAP分析...")
        try:
            # 调用本文件第151行定义的shap_analysis函数
            shap_analysis(
                model_wrapper, sample_texts,
                save_path=os.path.join(explain_dir, "shap_analysis.png")
            )
        except Exception as e:
            print(f"SHAP分析出错: {e}")
            print("提示: SHAP分析可能需要较长时间，可以尝试减少样本数量")
    
    # LIME分析
    if Config.USE_LIME:
        print("\n进行LIME分析...")
        try:
            # 调用本文件第185行定义的lime_analysis函数
            lime_analysis(
                model_wrapper, sample_texts,
                save_dir=explain_dir
            )
        except Exception as e:
            print(f"LIME分析出错: {e}")
    
    print(f"\n所有可解释性分析结果已保存到: {explain_dir}")


if __name__ == "__main__":
    model_path = os.path.join(Config.SAVE_MODEL_DIR, "final_model")
    if not os.path.exists(model_path):
        print(f"模型路径不存在: {model_path}")
        print("请先运行 train.py 训练模型")
    else:
        analyze_explainability(model_path, Config.DATASET_NAME)

