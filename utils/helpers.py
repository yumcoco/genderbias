# @Versin:1.0
# @Author:Yummy
"""
辅助函数工具库
包含文本处理、数据验证、格式化等通用功能
"""

import re
import json
import os
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from textblob import TextBlob


def load_bias_words() -> Dict:
    """加载偏向词汇字典"""
    try:
        assets_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets')
        with open(os.path.join(assets_path, 'bias_words.json'), 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("警告: bias_words.json 文件未找到，使用默认词汇")
        return get_default_bias_words()


def get_default_bias_words() -> Dict:
    """返回默认的偏向词汇"""
    return {
        "masculine_coded": {
            "competitive_terms": ["aggressive", "competitive", "dominant"],
            "tech_slang": ["ninja", "rockstar", "guru"]
        },
        "feminine_coded": {
            "collaborative_terms": ["collaborative", "supportive", "team-oriented"]
        },
        "inclusive_terms": {
            "diversity_words": ["diverse", "inclusive", "welcoming"]
        }
    }


def clean_text(text: str) -> str:
    """清理和标准化文本"""
    if not isinstance(text, str):
        return ""

    # 移除HTML标签
    text = re.sub(r'<[^>]+>', '', text)

    # 移除多余的空白字符
    text = re.sub(r'\s+', ' ', text)

    # 移除特殊字符（保留基本标点）
    text = re.sub(r'[^\w\s\.,!?;:-]', '', text)

    return text.strip()


def tokenize_text(text: str) -> List[str]:
    """文本分词"""
    # 转换为小写并分割
    words = re.findall(r'\b\w+\b', text.lower())
    return words


def count_word_matches(text: str, word_list: List[str]) -> Tuple[int, List[str]]:
    """计算文本中匹配的词汇数量"""
    words = tokenize_text(text)
    matches = []

    for word in word_list:
        if word.lower() in words:
            matches.append(word)

    return len(matches), matches


def calculate_sentiment(text: str) -> float:
    """计算文本情感分数 (-1到1)"""
    try:
        blob = TextBlob(text)
        return blob.sentiment.polarity
    except:
        return 0.0


def get_text_statistics(text: str) -> Dict:
    """获取文本统计信息"""
    words = tokenize_text(text)
    sentences = re.split(r'[.!?]+', text)

    return {
        'word_count': len(words),
        'sentence_count': len([s for s in sentences if s.strip()]),
        'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
        'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
        'character_count': len(text),
        'paragraph_count': len([p for p in text.split('\n\n') if p.strip()])
    }


def format_score(score: float, precision: int = 1) -> str:
    """格式化分数显示"""
    return f"{score:.{precision}f}"


def get_score_color(score: float) -> str:
    """根据分数获取颜色代码"""
    if score >= 80:
        return "#4CAF50"  # 绿色
    elif score >= 65:
        return "#8BC34A"  # 浅绿色
    elif score >= 50:
        return "#FFC107"  # 黄色
    elif score >= 35:
        return "#FF9800"  # 橙色
    else:
        return "#F44336"  # 红色


def get_score_label(score: float) -> str:
    """根据分数获取等级标签"""
    if score >= 80:
        return "优秀"
    elif score >= 65:
        return "良好"
    elif score >= 50:
        return "一般"
    elif score >= 35:
        return "较差"
    else:
        return "很差"


def extract_requirements(text: str) -> List[str]:
    """提取职位要求列表"""
    # 查找常见的要求表达模式
    patterns = [
        r'(?:requirements?|qualifications?|必备条件|任职要求)[:\s]*(.+?)(?=\n\n|\n[A-Z]|$)',
        r'(?:必须|required|must have)[:\s]*(.+?)(?=\n|\.|;)',
        r'(?:skills?|技能)[:\s]*(.+?)(?=\n\n|\n[A-Z]|$)'
    ]

    requirements = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            # 分割成单独的要求
            items = re.split(r'[•\n\r\t;]', match)
            requirements.extend([item.strip() for item in items if item.strip()])

    return requirements[:10]  # 限制最多10个要求


def validate_text_input(text: str, min_length: int = 50, max_length: int = 5000) -> Tuple[bool, str]:
    """验证文本输入"""
    if not text or not text.strip():
        return False, "请输入职位描述内容"

    if len(text) < min_length:
        return False, f"文本过短，至少需要{min_length}个字符"

    if len(text) > max_length:
        return False, f"文本过长，最多支持{max_length}个字符"

    return True, "验证通过"


def load_dataset(file_path: str) -> Optional[pd.DataFrame]:
    """安全加载数据集"""
    try:
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return None

        df = pd.read_csv(file_path)
        print(f"成功加载数据集: {file_path} ({len(df)} 行)")
        return df

    except Exception as e:
        print(f"加载数据集失败: {e}")
        return None


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """安全除法，避免除零错误"""
    try:
        return numerator / denominator if denominator != 0 else default
    except:
        return default


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """截断文本显示"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def highlight_words(text: str, words: List[str], color: str = "yellow") -> str:
    """在文本中高亮显示指定词汇（返回HTML）"""
    highlighted_text = text
    for word in words:
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        highlighted_text = pattern.sub(
            f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px;">{word}</span>',
            highlighted_text
        )
    return highlighted_text