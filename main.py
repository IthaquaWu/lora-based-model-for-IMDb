# -*- coding: utf-8 -*-
"""
主运行脚本：整合训练、评估和可解释性分析

主要功能：
- main(): 统一入口，支持训练、评估、可解释性分析的单独或组合运行

调用关系：
- 调用 train.train() (train.py第42行) 进行模型训练
- 调用 evaluate.evaluate_model() (evaluate.py第18行) 进行模型评估
- 调用 explainability.analyze_explainability() (explainability.py第228行) 进行可解释性分析
- 调用 ensemble.py 中的函数进行模型集成
"""
import argparse
import os
import torch
from config import Config  # 配置文件
from train import train  # 训练模块，定义在train.py第42行
from evaluate import evaluate_model  # 评估模块，定义在evaluate.py第18行
from explainability import analyze_explainability  # 可解释性模块，定义在explainability.py第228行
from ensemble import ensemble_predict, load_ensemble_models  # 模型集成模块


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='基于DistilBERT的文本分类与可解释性分析')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['train', 'evaluate', 'explain', 'ensemble', 'all'],
                       help='运行模式: train(训练), evaluate(评估), explain(可解释性), ensemble(模型集成), all(全部)')
    parser.add_argument('--model_path', type=str, default=None,
                       help='模型路径（用于评估和可解释性分析）')
    parser.add_argument('--ensemble_model_paths', type=str, default=None,
                       help='要集成的模型路径列表，用逗号分隔')
    parser.add_argument('--dataset', type=str, default='imdb',
                       choices=['imdb'],
                       help='数据集名称')
    parser.add_argument('--use_data_augmentation', type=str, default='auto',
                       choices=['auto', 'true', 'false'],
                       help='是否使用数据增强: auto(根据config.py配置), true(启用), false(禁用)')
    
    args = parser.parse_args()
    
    # 创建必要的目录
    Config.create_dirs()
    
    # 设置数据集
    Config.DATASET_NAME = args.dataset
    
    # 设置数据增强
    if args.use_data_augmentation.lower() == 'true':
        Config.USE_DATA_AUGMENTATION = True
    elif args.use_data_augmentation.lower() == 'false':
        Config.USE_DATA_AUGMENTATION = False
    # 如果是'auto'，则保持Config中的默认设置
    
    if args.mode == 'train' or args.mode == 'all':
        print("="*50)
        print("开始训练模型...")
        print("="*50)
        train()  # 调用train.py第42行定义的train函数
    
    if args.mode == 'evaluate' or args.mode == 'all':
        print("\n" + "="*50)
        print("开始评估模型...")
        print("="*50)
        model_path = args.model_path or os.path.join(Config.SAVE_MODEL_DIR, "final_model")
        if not os.path.exists(model_path):
            print(f"警告: 模型路径不存在 {model_path}")
            print("请先运行训练模式")
        else:
            # 调用evaluate.py第18行定义的evaluate_model函数
            evaluate_model(model_path, args.dataset)
    
    if args.mode == 'explain' or args.mode == 'all':
        print("\n" + "="*50)
        print("开始可解释性分析...")
        print("="*50)
        model_path = args.model_path or os.path.join(Config.SAVE_MODEL_DIR, "final_model")
        if not os.path.exists(model_path):
            print(f"警告: 模型路径不存在 {model_path}")
            print("请先运行训练模式")
        else:
            # 调用explainability.py第228行定义的analyze_explainability函数
            analyze_explainability(model_path, args.dataset)

    if args.mode == 'ensemble' or args.mode == 'all':
        print("\n" + "="*50)
        print("开始模型集成...")
        print("="*50)
        
        if args.ensemble_model_paths:
            # 将逗号分隔的模型路径转换为列表
            model_paths = [path.strip() for path in args.ensemble_model_paths.split(',')]
            
            # 检查模型路径是否存在
            valid_paths = []
            for path in model_paths:
                if os.path.exists(path):
                    valid_paths.append(path)
                else:
                    print(f"警告: 模型路径不存在 {path}")
            
            if valid_paths:
                # 加载分词器
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
                
                # 设置设备
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print(f"使用设备: {device}")
                
                # 加载模型集成
                models = load_ensemble_models(valid_paths, device)
                
                if models:
                    print(f"成功加载 {len(models)} 个模型用于集成")
                    
                    # 简单的测试示例
                    test_text = "This movie is absolutely fantastic! The acting was brilliant and the plot kept me engaged throughout."
                    print(f"\n测试文本: {test_text}")
                    
                    # 执行集成预测
                    label, probs = ensemble_predict(models, tokenizer, test_text, device)
                    
                    print(f"集成预测结果: {'正面' if label == 1 else '负面'} (概率: {probs[label]:.4f})")
                    print(f"完整概率分布: 负面: {probs[0]:.4f}, 正面: {probs[1]:.4f}")
                else:
                    print("没有成功加载任何模型用于集成")
            else:
                print("没有有效的模型路径用于集成")
        else:
            print("请使用 --ensemble_model_paths 参数指定要集成的模型路径")
            print("示例: python main.py --mode ensemble --ensemble_model_paths path/to/model1.bin,path/to/model2.bin")

    print("\n" + "="*50)
    print("所有任务完成！")
    print("="*50)


if __name__ == "__main__":
    main()


