"""主程序脚本：测试数据加载与预处理流程"""
import logging
import torch
from torch.utils.data import DataLoader
from data_loader import prepare_datasets
from config import Config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def main():
    """主函数：执行数据加载与预处理测试"""
    try:
        # 1. 打印配置信息
        logger.info("===== 配置信息 =====")
        logger.info(f"使用模型: {Config.MODEL_NAME}")
        logger.info(f"数据集路径: {Config.DATASET_PATH}")
        logger.info(f"最大序列长度: {Config.MAX_LENGTH}")
        logger.info(f"验证集划分比例: {Config.VALIDATION_SPLIT}")
        logger.info(f"是否使用本地数据集: {Config.USE_LOCAL_DATASET}")
        logger.info("====================")

        # 2. 准备数据集（加载+预处理）
        logger.info("开始加载并预处理数据集...")
        tokenizer, train_dataset, val_dataset, test_dataset = prepare_datasets(
            dataset_name=Config.DATASET_NAME  # 从配置文件读取数据集名称
        )

        # 3. 测试数据集基本信息
        logger.info("\n===== 数据集基本信息 =====")
        logger.info(f"训练集样本数: {len(train_dataset)}")
        logger.info(f"验证集样本数: {len(val_dataset)}")
        logger.info(f"测试集样本数: {len(test_dataset)}")

        # 4. 随机抽取样本查看预处理效果
        logger.info("\n===== 样本预处理效果 =====")
        sample_idx = 0  # 查看第0个样本
        sample = train_dataset[sample_idx]

        # 打印原始文本（截断显示）
        raw_text = train_dataset.texts[sample_idx] if hasattr(train_dataset, 'texts') else "无法获取原始文本"
        logger.info(f"原始文本（前200字符）: {raw_text[:200]}...")

        # 打印编码后信息
        logger.info(f"编码后input_ids长度: {len(sample['input_ids'])}")
        logger.info(f"注意力掩码长度: {len(sample['attention_mask'])}")
        logger.info(f"样本标签: {sample['labels'].item()}")

        # 解码回文本查看分词效果
        decoded_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
        logger.info(f"解码后文本（前200字符）: {decoded_text[:200]}...")

        # 5. 测试DataLoader（模拟训练时的数据加载）
        logger.info("\n===== 测试DataLoader =====")
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS
        )

        # 获取一个批次的数据
        batch = next(iter(train_loader))
        logger.info(f"批次数据结构: {list(batch.keys())}")
        logger.info(f"批次input_ids形状: {batch['input_ids'].shape}")  # (batch_size, max_length)
        logger.info(f"批次labels形状: {batch['labels'].shape}")  # (batch_size,)

        logger.info("\n所有测试完成，数据加载与预处理流程正常！")

    except Exception as e:
        logger.error(f"执行过程中出错: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()