"""
性别偏向检测器
识别职位描述中的性别偏向词汇和模式
"""

import re
import sys
import os
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from datasets import load_dataset
from transformers import pipeline
# from core.bias_detector_tune import fine_tune_md_gender_bias  # 自己的finetune函数

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import MASCULINE_WORDS, FEMININE_WORDS, INCLUSIVE_WORDS, EXCLUSIVE_WORDS
from utils.helpers import clean_text, tokenize_text, count_word_matches, load_bias_words


@dataclass
class BiasAnalysisResult:
    """偏向分析结果数据类"""
    masculine_words: List[str]
    feminine_words: List[str]
    inclusive_words: List[str]
    exclusive_words: List[str]
    masculine_score: float
    feminine_score: float
    inclusive_score: float
    exclusive_score: float
    overall_bias: str  # 'masculine', 'feminine', 'neutral'
    bias_strength: float  # 0-1, 偏向强度


class BiasDetector:
    """性别偏向检测器主类"""

    def __init__(self, use_hf: bool = False, use_model: bool = True):
        """初始化检测器，支持加载本地或Hugging Face词表"""
        if use_hf:
            print("🔗 Loading bias lexicon from Hugging Face...")
            self.bias_words_dict = self._load_hf_bias_lexicon("facebook/md_gender_bias")
        else:
            self.bias_words_dict = load_bias_words()

        self.masculine_words = self._extract_words_from_dict('masculine_coded')
        self.feminine_words = self._extract_words_from_dict('feminine_coded')
        self.inclusive_words = self._extract_words_from_dict('inclusive_terms')
        self.exclusive_words = self._extract_words_from_dict('exclusive_indicators')

        self.use_model = use_model
        self.classifier = None

        # if self.use_model:
        #     print("🔍 Loading fine-tuned MD gender bias classifier...")
        #     model, tokenizer = fine_tune_md_gender_bias()
        #     self.classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

        if self.use_model:
            print("⚠️ Model loading skipped: fine-tune function not implemented.")

        print(f"Bias Detector initialized successfully")
        print(f"   Masculine words: {len(self.masculine_words)}")
        print(f"   Feminine words: {len(self.feminine_words)}")
        print(f"   Inclusive words: {len(self.inclusive_words)}")
        print(f"   Exclusive words: {len(self.exclusive_words)}")

    def _load_hf_bias_lexicon(self, dataset_name: str) -> dict:
        """从 Hugging Face 数据集中加载词汇表"""
        dataset = load_dataset(dataset_name, split='train', trust_remote_code=True)
        word_dict = {'masculine_coded': [], 'feminine_coded': [], 'inclusive_terms': [], 'exclusive_indicators': []}

        for entry in dataset:
            category = entry.get("bias_type", "")
            term = entry.get("term", "").strip().lower()
            if "masculine" in category:
                word_dict['masculine_coded'].append(term)
            elif "feminine" in category:
                word_dict['feminine_coded'].append(term)
            elif "inclusive" in category:
                word_dict['inclusive_terms'].append(term)
            elif "exclusive" in category:
                word_dict['exclusive_indicators'].append(term)

        return word_dict

    def _extract_words_from_dict(self, category: str) -> List[str]:
        """从字典中提取指定类别的词汇列表"""
        words = []
        if category in self.bias_words_dict:
            category_data = self.bias_words_dict[category]
            if isinstance(category_data, dict):
                for subcategory, word_list in category_data.items():
                    if isinstance(word_list, list):
                        words.extend(word_list)
            elif isinstance(category_data, list):
                words.extend(category_data)

        # 去重并转为小写
        return list(set([word.lower() for word in words]))

    def detect_bias_words(self, text: str) -> Dict[str, List[str]]:
        """检测文本中的各类偏向词汇"""
        cleaned_text = clean_text(text)

        # 检测各类词汇
        masc_count, masc_matches = count_word_matches(cleaned_text, self.masculine_words)
        fem_count, fem_matches = count_word_matches(cleaned_text, self.feminine_words)
        incl_count, incl_matches = count_word_matches(cleaned_text, self.inclusive_words)
        excl_count, excl_matches = count_word_matches(cleaned_text, self.exclusive_words)

        return {
            'masculine': masc_matches,
            'feminine': fem_matches,
            'inclusive': incl_matches,
            'exclusive': excl_matches,
            'counts': {
                'masculine': masc_count,
                'feminine': fem_count,
                'inclusive': incl_count,
                'exclusive': excl_count
            }
        }

    def calculate_bias_scores(self, text: str) -> Dict[str, float]:
        """计算各维度的偏向评分（每100词的密度）"""
        bias_words = self.detect_bias_words(text)
        text_length = len(tokenize_text(text))

        # 避免除零错误
        if text_length == 0:
            return {'masculine': 0, 'feminine': 0, 'inclusive': 0, 'exclusive': 0}

        # 计算相对密度 (每100词的出现次数)
        scores = {}
        for category in ['masculine', 'feminine', 'inclusive', 'exclusive']:
            count = bias_words['counts'][category]
            scores[category] = (count / text_length) * 100

        return scores

    def analyze_bias_patterns(self, text: str) -> BiasAnalysisResult:
        """全面分析文本的偏向模式"""
        bias_words = self.detect_bias_words(text)
        bias_scores = self.calculate_bias_scores(text)

        # 计算整体偏向趋势
        masculine_strength = bias_scores['masculine'] - bias_scores['feminine']
        overall_bias = 'neutral'
        bias_strength = 0.0

        if masculine_strength > 2:
            overall_bias = 'masculine'
            bias_strength = min(masculine_strength / 10, 1.0)
        elif masculine_strength < -2:
            overall_bias = 'feminine'
            bias_strength = min(abs(masculine_strength) / 10, 1.0)
        else:
            bias_strength = abs(masculine_strength) / 10

        return BiasAnalysisResult(
            masculine_words=bias_words['masculine'],
            feminine_words=bias_words['feminine'],
            inclusive_words=bias_words['inclusive'],
            exclusive_words=bias_words['exclusive'],
            masculine_score=bias_scores['masculine'],
            feminine_score=bias_scores['feminine'],
            inclusive_score=bias_scores['inclusive'],
            exclusive_score=bias_scores['exclusive'],
            overall_bias=overall_bias,
            bias_strength=bias_strength
        )

    def get_improvement_suggestions(self, analysis: BiasAnalysisResult) -> List[str]:
        """基于分析结果生成英文改进建议"""
        suggestions = []

        # 男性化词汇过多的建议
        if analysis.masculine_score > 5:
            suggestions.append(
                f"Detected {len(analysis.masculine_words)} masculine-coded words. "
                f"Consider replacing with neutral alternatives: {', '.join(analysis.masculine_words[:3])}"
            )

        # 包容性词汇不足的建议
        if analysis.inclusive_score < 2:
            suggestions.append(
                "Add more inclusive language such as 'diversity', 'flexibility', 'career development', 'work-life balance'"
            )

        # 排他性词汇过多的建议
        if analysis.exclusive_score > 3:
            suggestions.append(
                f"Detected {len(analysis.exclusive_words)} exclusive expressions. "
                f"Consider softer language: {', '.join(analysis.exclusive_words[:2])}"
            )

        # 整体偏向建议
        if analysis.overall_bias == 'masculine' and analysis.bias_strength > 0.3:
            suggestions.append(
                "Job description leans masculine. Consider adding collaborative and supportive language"
            )

        # 平衡性建议
        if len(analysis.feminine_words) == 0 and len(analysis.masculine_words) > 0:
            suggestions.append(
                "Consider balancing with words emphasizing teamwork and communication"
            )

        if not suggestions:
            suggestions.append("Gender expression in job description is relatively balanced. Keep it up!")

        return suggestions

    def detect_problematic_phrases(self, text: str) -> List[Dict[str, str]]:
        """检测有问题的短语模式"""
        problematic_patterns = [
            {
                'pattern': r'\b(?:must\s+have|required|essential|mandatory)\b.*?\b(?:years?|experience)\b',
                'type': 'strict_requirement',
                'description': 'Overly strict experience requirements'
            },
            {
                'pattern': r'\b(?:ninja|rockstar|guru|wizard|champion)\b',
                'type': 'tech_slang',
                'description': 'Masculine-coded tech industry slang'
            },
            {
                'pattern': r'\b(?:aggressive|dominant|forceful)\s+(?:\w+\s+)*(?:approach|style|personality)\b',
                'type': 'aggressive_trait',
                'description': 'Emphasis on aggressive traits'
            },
            {
                'pattern': r'\b(?:fast[-\s]paced|high[-\s]pressure|demanding)\s+environment\b',
                'type': 'pressure_environment',
                'description': 'High-pressure work environment description'
            }
        ]

        found_phrases = []
        text_lower = text.lower()

        for pattern_info in problematic_patterns:
            matches = re.findall(pattern_info['pattern'], text_lower, re.IGNORECASE)
            for match in matches:
                found_phrases.append({
                    'phrase': match,
                    'type': pattern_info['type'],
                    'description': pattern_info['description']
                })

        return found_phrases

    def generate_detailed_report(self, text: str) -> Dict[str, Any]:
        """生成详细的偏向分析报告"""
        analysis = self.analyze_bias_patterns(text)
        suggestions = self.get_improvement_suggestions(analysis)
        problematic_phrases = self.detect_problematic_phrases(text)

        return {
            'analysis': analysis,
            'suggestions': suggestions,
            'problematic_phrases': problematic_phrases,
            'summary': {
                'total_bias_words': (
                        len(analysis.masculine_words) +
                        len(analysis.feminine_words) +
                        len(analysis.exclusive_words)
                ),
                'bias_direction': analysis.overall_bias,
                'bias_strength': analysis.bias_strength,
                'inclusivity_level': 'high' if analysis.inclusive_score > 3 else
                'medium' if analysis.inclusive_score > 1 else 'low'
            }
        }
    
# 若使用 Hugging Face 词表，可创建时传入 use_hf=True
# detector = BiasDetector(use_hf=True)


# 创建全局检测器实例
bias_detector = BiasDetector()


def get_bias_detector() -> BiasDetector:
    """获取偏向检测器实例"""
    return bias_detector

