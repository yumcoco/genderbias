"""
Bias detector@P4G
"""

import re
import sys
import os
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import MASCULINE_WORDS, FEMININE_WORDS, INCLUSIVE_WORDS, EXCLUSIVE_WORDS
from utils.helpers import clean_text, tokenize_text, count_word_matches, load_bias_words


@dataclass
class BiasAnalysisResult:
    masculine_words: List[str]
    feminine_words: List[str]
    inclusive_words: List[str]
    exclusive_words: List[str]
    masculine_score: float
    feminine_score: float
    inclusive_score: float
    exclusive_score: float
    overall_bias: str  # 'masculine', 'feminine', 'neutral'
    bias_strength: float


class BiasDetector:

    def __init__(self):
        self.bias_words_dict = load_bias_words()
        self.masculine_words = self._extract_words_from_dict('masculine_coded')
        self.feminine_words = self._extract_words_from_dict('feminine_coded')
        self.inclusive_words = self._extract_words_from_dict('inclusive_terms')
        self.exclusive_words = self._extract_words_from_dict('exclusive_indicators')

        print(f"Bias Detector initialized successfully")
        print(f"   Masculine words: {len(self.masculine_words)}")
        print(f"   Feminine words: {len(self.feminine_words)}")
        print(f"   Inclusive words: {len(self.inclusive_words)}")
        print(f"   Exclusive words: {len(self.exclusive_words)}")

    def _extract_words_from_dict(self, category: str) -> List[str]:
        words = []
        if category in self.bias_words_dict:
            category_data = self.bias_words_dict[category]
            if isinstance(category_data, dict):
                for subcategory, word_list in category_data.items():
                    if isinstance(word_list, list):
                        words.extend(word_list)
            elif isinstance(category_data, list):
                words.extend(category_data)

        return list(set([word.lower() for word in words]))

    def detect_bias_words(self, text: str) -> Dict[str, List[str]]:
        cleaned_text = clean_text(text)

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
        bias_words = self.detect_bias_words(text)
        text_length = len(tokenize_text(text))

        if text_length == 0:
            return {'masculine': 0, 'feminine': 0, 'inclusive': 0, 'exclusive': 0}

        scores = {}
        for category in ['masculine', 'feminine', 'inclusive', 'exclusive']:
            count = bias_words['counts'][category]
            scores[category] = (count / text_length) * 100

        return scores

    def analyze_bias_patterns(self, text: str) -> BiasAnalysisResult:
        bias_words = self.detect_bias_words(text)
        bias_scores = self.calculate_bias_scores(text)

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
        suggestions = []


        if analysis.masculine_score > 5:
            suggestions.append(
                f"Detected {len(analysis.masculine_words)} masculine-coded words. "
                f"Consider replacing with neutral alternatives: {', '.join(analysis.masculine_words[:3])}"
            )

        if analysis.inclusive_score < 2:
            suggestions.append(
                "Add more inclusive language such as 'diversity', 'flexibility', 'career development', 'work-life balance'"
            )

        if analysis.exclusive_score > 3:
            suggestions.append(
                f"Detected {len(analysis.exclusive_words)} exclusive expressions. "
                f"Consider softer language: {', '.join(analysis.exclusive_words[:2])}"
            )

        if analysis.overall_bias == 'masculine' and analysis.bias_strength > 0.3:
            suggestions.append(
                "Job description leans masculine. Consider adding collaborative and supportive language"
            )

        if len(analysis.feminine_words) == 0 and len(analysis.masculine_words) > 0:
            suggestions.append(
                "Consider balancing with words emphasizing teamwork and communication"
            )

        if not suggestions:
            suggestions.append("Gender expression in job description is relatively balanced. Keep it up!")

        return suggestions

    def detect_problematic_phrases(self, text: str) -> List[Dict[str, str]]:
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


bias_detector = BiasDetector()


def get_bias_detector() -> BiasDetector:
    return bias_detector