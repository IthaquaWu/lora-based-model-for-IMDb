"""文本数据加载与预处理模块"""
import os
import re
import string
import random
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from transformers import AutoTokenizer
from typing import Tuple, List, Dict, Any
import nltk
from nltk.corpus import wordnet

# 导入配置类
from config import Config

# 下载nltk资源（如果需要）
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# 扩展标点符号集合（包含常见特殊字符）
EXTENDED_PUNCTUATION = string.punctuation + "’‘“”…—–"
# 情感相关标点符号
EMOTIONAL_PUNCTUATION = "!?！？"


class TextClassificationDataset(Dataset):
    """文本分类数据集类"""

    def __init__(self, texts, labels, tokenizer, max_length=512, clean_text=True):
        """
        初始化数据集

        Args:
            texts: 文本列表
            labels: 标签列表
            tokenizer: 分词器
            max_length: 最大序列长度
            clean_text: 是否对文本进行清洗
        """
        self.texts = [self.clean_text(text) if clean_text else text for text in texts]
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._validate_data()  # 验证数据有效性

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """
        获取单个样本

        Args:
            idx: 样本索引

        Returns:
            dict: 包含input_ids, attention_mask, labels的字典
        """
        text = str(self.texts[idx])
        label = self.labels[idx]

        # 对文本进行编码
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
        }

    @staticmethod
    def clean_text(text: str) -> str:
        """
        文本清洗函数：标准化文本格式，移除噪声，但保留情感相关标点

        Args:
            text: 原始文本

        Returns:
            清洗后的文本
        """
        if not isinstance(text, str):
            text = str(text)

        # 1. 转换为小写
        text = text.lower()

        # 2. 移除HTML标签
        text = re.sub(r'<[^>]*?>', ' ', text)

        # 3. 移除URLs
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

        # 4. 移除电子邮件
        text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w{2,4}\b', ' ', text)

        # 5. 移除数字（保留文本特征）
        text = re.sub(r'\d+', ' ', text)

        # 6. 仅保留情感相关标点符号，移除其他标点
        def keep_emotional_punc(char):
            return char if char in EMOTIONAL_PUNCTUATION or char.isalnum() or char.isspace() else ' '
        text = ''.join([keep_emotional_punc(c) for c in text])

        # 7. 处理空白字符
        text = re.sub(r'\s+', ' ', text).strip()

        # 8. 移除太短的文本（少于3个字符）
        if len(text) < 3:
            return ""

        return text

    def _validate_data(self):
        """验证数据集的一致性和有效性"""
        # 检查文本和标签长度是否一致
        if len(self.texts) != len(self.labels):
            raise ValueError(f"文本数量 ({len(self.texts)}) 与标签数量 ({len(self.labels)}) 不匹配")

        # 检查空文本
        empty_text_indices = [i for i, text in enumerate(self.texts) if not text.strip()]
        if empty_text_indices:
            print(f"警告: 发现 {len(empty_text_indices)} 个空文本样本，已自动过滤")
            # 过滤空文本
            self.texts = [t for i, t in enumerate(self.texts) if i not in empty_text_indices]
            self.labels = [l for i, l in enumerate(self.labels) if i not in empty_text_indices]

        # 检查标签是否为整数
        try:
            self.labels = [int(l) for l in self.labels]
        except ValueError:
            raise ValueError("标签必须是整数类型")


def load_imdb_dataset() -> Tuple[Tuple[List[str], List[int]], Tuple[List[str], List[int]]]:
    """
    加载IMDb电影评论数据集（从本地CSV文件）

    Returns:
        train_dataset, test_dataset: 训练集和测试集 (文本列表, 标签列表)
    """
    if Config.USE_LOCAL_DATASET:
        print("正在从本地加载IMDb数据集...")
        train_path = os.path.join(Config.DATASET_PATH, Config.TRAIN_FILE)
        test_path = os.path.join(Config.DATASET_PATH, Config.TEST_FILE)

        # 检查文件是否存在
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"训练集文件不存在: {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"测试集文件不存在: {test_path}")

        # 读取CSV文件（支持不同编码）
        encodings = ['utf-8', 'latin-1', 'utf-16']
        train_df = None
        test_df = None

        for encoding in encodings:
            try:
                if train_df is None:
                    train_df = pd.read_csv(train_path, encoding=encoding)
                if test_df is None:
                    test_df = pd.read_csv(test_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue

        if train_df is None or test_df is None:
            raise UnicodeDecodeError("无法解析CSV文件，请检查文件编码")

        # 验证必要的列是否存在
        required_columns = ['text', 'label']
        for col in required_columns:
            if col not in train_df.columns:
                raise ValueError(f"训练集缺少必要的列: {col}")
            if col not in test_df.columns:
                raise ValueError(f"测试集缺少必要的列: {col}")

        # 处理缺失值
        train_df = train_df.dropna(subset=required_columns)
        test_df = test_df.dropna(subset=required_columns)

        print(f"加载原始训练集: {len(train_df)} 样本")
        print(f"加载原始测试集: {len(test_df)} 样本")

        # 提取文本和标签
        train_texts = train_df['text'].tolist()
        train_labels = train_df['label'].tolist()
        test_texts = test_df['text'].tolist()
        test_labels = test_df['label'].tolist()

        return (train_texts, train_labels), (test_texts, test_labels)
    else:
        # 从Hugging Face加载（备用方案）
        print("正在从Hugging Face加载IMDb数据集...")
        from datasets import load_dataset
        dataset = load_dataset("imdb")

        train_texts = dataset['train']['text'].tolist()
        train_labels = dataset['train']['label'].tolist()
        test_texts = dataset['test']['text'].tolist()
        test_labels = dataset['test']['label'].tolist()

        print(f"训练集大小: {len(train_texts)}")
        print(f"测试集大小: {len(test_texts)}")

        return (train_texts, train_labels), (test_texts, test_labels)


def load_thucnews_dataset() -> Tuple[Tuple[List[str], List[int]], Tuple[List[str], List[int]]]:
    """
    加载THUCNews中文新闻数据集
    注意：需要先下载THUCNews数据集到本地

    Returns:
        train_dataset, test_dataset: 训练集和测试集
    """
    print("正在加载THUCNews数据集...")
    if Config.USE_LOCAL_DATASET:
        train_path = os.path.join(Config.DATASET_PATH, Config.TRAIN_FILE)
        test_path = os.path.join(Config.DATASET_PATH, Config.TEST_FILE)

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError("THUCNews数据集文件不存在，请检查路径配置")

        # 读取CSV文件
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # 验证必要的列
        required_columns = ['text', 'label']
        for col in required_columns:
            if col not in train_df.columns or col not in test_df.columns:
                raise ValueError(f"THUCNews数据集缺少必要的列: {col}")

        # 处理缺失值
        train_df = train_df.dropna(subset=required_columns)
        test_df = test_df.dropna(subset=required_columns)

        # 提取文本和标签
        train_texts = train_df['text'].tolist()
        train_labels = train_df['label'].tolist()
        test_texts = test_df['text'].tolist()
        test_labels = test_df['label'].tolist()

        print(f"训练集大小: {len(train_texts)}")
        print(f"测试集大小: {len(test_texts)}")

        return (train_texts, train_labels), (test_texts, test_labels)
    else:
        raise NotImplementedError("THUCNews数据集暂不支持从Hugging Face加载，请使用本地数据集")


def prepare_datasets(dataset_name="imdb", data_augmentation=False, augment_rate=0.3) -> Tuple[AutoTokenizer, Dataset, Dataset, Dataset]:
    """
    准备数据集，包括加载、清洗和预处理

    Args:
        dataset_name: 数据集名称
        data_augmentation: 是否进行数据增强
        augment_rate: 数据增强比例（0-1）

    Returns:
        tokenizer, train_dataset, val_dataset, test_dataset: 分词器和数据集
    """
    # 加载分词器
    print(f"正在加载分词器: {Config.MODEL_NAME}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, local_files_only=True)
        print("从本地加载分词器成功")
    except:
        print("本地未找到分词器，从Hugging Face加载...")
        tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # 加载数据集
    if dataset_name.lower() == "imdb":
        (train_texts, train_labels), (test_texts, test_labels) = load_imdb_dataset()

        # 检查是否有本地验证集
        val_texts, val_labels = None, None
        valid_path = os.path.join(Config.DATASET_PATH, Config.VALID_FILE)
        if Config.USE_LOCAL_DATASET and os.path.exists(valid_path):
            print(f"读取验证集: {valid_path}")
            valid_df = pd.read_csv(valid_path)
            if 'text' in valid_df.columns and 'label' in valid_df.columns:
                valid_df = valid_df.dropna(subset=['text', 'label'])
                val_texts = valid_df['text'].tolist()
                val_labels = valid_df['label'].tolist()
                print(f"验证集大小: {len(val_texts)}")
            else:
                print("验证集格式不正确，将从训练集划分")

    elif dataset_name.lower() == "thucnews":
        (train_texts, train_labels), (test_texts, test_labels) = load_thucnews_dataset()
        val_texts, val_labels = None, None
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    # 应用数据增强
    if data_augmentation:
        print(f"正在应用数据增强，增强比例: {augment_rate}")
        num_augmented = int(len(train_texts) * augment_rate)
        augmented_texts = []
        augmented_labels = []
        
        # 随机选择要增强的样本
        indices = random.sample(range(len(train_texts)), num_augmented)
        
        for idx in indices:
            original_text = train_texts[idx]
            label = train_labels[idx]
            
            # 对每个样本生成多个增强版本
            augmented_versions = augment_text(original_text, num_augments=2)
            for augmented_text in augmented_versions:
                augmented_texts.append(augmented_text)
                augmented_labels.append(label)
        
        # 将增强数据与原始数据合并
        train_texts.extend(augmented_texts)
        train_labels.extend(augmented_labels)
        print(f"增强后训练集大小: {len(train_texts)} (新增{len(augmented_texts)}个样本)")

    # 创建数据集对象（包含文本清洗）
    train_dataset = TextClassificationDataset(
        train_texts, train_labels, tokenizer, Config.MAX_LENGTH, clean_text=True
    )
    test_dataset = TextClassificationDataset(
        test_texts, test_labels, tokenizer, Config.MAX_LENGTH, clean_text=True
    )

    # 处理验证集
    if val_texts is not None and val_labels is not None:
        val_dataset = TextClassificationDataset(
            val_texts, val_labels, tokenizer, Config.MAX_LENGTH, clean_text=True
        )
    else:
        # 从训练集中划分验证集
        train_size = int((1 - Config.VALIDATION_SPLIT) * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

    print(f"预处理后数据集大小:")
    print(f"  训练集: {len(train_dataset)}")
    print(f"  验证集: {len(val_dataset)}")
    print(f"  测试集: {len(test_dataset)}")

    return tokenizer, train_dataset, val_dataset, test_dataset


# 数据增强函数

def get_synonyms(word):
    """
    获取单词的同义词列表
    
    Args:
        word: 原始单词
        
    Returns:
        同义词列表
    """
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym != word:
                synonyms.add(synonym)
    return list(synonyms)


def synonym_replacement(text, n=2):
    """
    同义词替换增强
    
    Args:
        text: 原始文本
        n: 替换的单词数量
        
    Returns:
        增强后的文本
    """
    words = text.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word.isalpha()]))
    random.shuffle(random_word_list)
    num_replaced = 0
    
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    
    return ' '.join(new_words)


def random_insertion(text, n=2):
    """
    随机插入增强
    
    Args:
        text: 原始文本
        n: 插入的单词数量
        
    Returns:
        增强后的文本
    """
    words = text.split()
    new_words = words.copy()
    
    for _ in range(n):
        add_word(new_words)
    
    return ' '.join(new_words)


def add_word(new_words):
    """
    向文本中随机插入同义词
    
    Args:
        new_words: 单词列表
    """
    synonyms = []
    counter = 0
    
    while len(synonyms) < 1:
        if counter >= len(new_words):
            random_word = random.choice(new_words)
        else:
            random_word = new_words[counter]
            counter += 1
        synonyms = get_synonyms(random_word)
    
    random_synonym = random.choice(synonyms)
    random_idx = random.randint(0, len(new_words) - 1)
    new_words.insert(random_idx, random_synonym)


def random_deletion(text, p=0.1):
    """
    随机删除增强
    
    Args:
        text: 原始文本
        p: 删除概率
        
    Returns:
        增强后的文本
    """
    words = text.split()
    
    if len(words) == 1:
        return text
    
    new_words = [word for word in words if random.uniform(0, 1) > p]
    
    if len(new_words) == 0:
        return random.choice(words)
    
    return ' '.join(new_words)


def random_swap(text, n=2):
    """
    随机交换增强
    
    Args:
        text: 原始文本
        n: 交换的次数
        
    Returns:
        增强后的文本
    """
    words = text.split()
    new_words = words.copy()
    
    for _ in range(n):
        new_words = swap_word(new_words)
    
    return ' '.join(new_words)


def swap_word(new_words):
    """
    随机交换两个单词的位置
    
    Args:
        new_words: 单词列表
        
    Returns:
        交换后的单词列表
    """
    if len(new_words) < 2:
        return new_words
    
    idx1, idx2 = random.sample(range(len(new_words)), 2)
    new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    return new_words


def augment_text(text, augmentation_methods=None, num_augments=1):
    """
    综合数据增强函数
    
    Args:
        text: 原始文本
        augmentation_methods: 使用的数据增强方法列表
        num_augments: 生成的增强文本数量
        
    Returns:
        增强后的文本列表
    """
    if augmentation_methods is None:
        augmentation_methods = [
            (synonym_replacement, {}),
            (random_insertion, {}),
            (random_deletion, {}),
            (random_swap, {})
        ]
    
    augmented_texts = []
    for _ in range(num_augments):
        # 随机选择一个增强方法
        method, params = random.choice(augmentation_methods)
        augmented_text = method(text, **params)
        augmented_texts.append(augmented_text)
    
    return augmented_texts


