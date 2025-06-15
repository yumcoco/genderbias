"""
Inclusivity_Scorer @ P4G
"""

import sys
import os
from typing import Dict, List, Tuple
import math
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import SCORING_WEIGHTS, SCORE_LEVELS
from utils.helpers import get_text_statistics, calculate_sentiment, get_score_color, get_score_label
from core.bias_detector import get_bias_detector


@dataclass
class InclusivityScore:

    overall_score: float
    component_scores: Dict[str, float]
    grade: str
    color: str
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]


class InclusivityScorer:

    def __init__(self):
        self.bias_detector = get_bias_detector()
        self.weights = SCORING_WEIGHTS
        self.score_levels = SCORE_LEVELS

        print("Inclusivity Scorer initialized successfully")

    def calculate_component_scores(self, text: str) -> Dict[str, float]:
        bias_analysis = self.bias_detector.analyze_bias_patterns(text)

        text_stats = get_text_statistics(text)

        sentiment_score = calculate_sentiment(text)

        scores = {}
        masculine_penalty = len(bias_analysis.masculine_words) * self.weights['masculine_penalty']
        feminine_bonus = len(bias_analysis.feminine_words) * self.weights['feminine_bonus']
        language_balance = 50 + feminine_bonus + masculine_penalty
        scores['language_balance'] = max(0, min(100, language_balance))

        inclusive_bonus = len(bias_analysis.inclusive_words) * self.weights['inclusive_bonus']
        inclusivity_base = 30 + inclusive_bonus
        scores['inclusivity'] = max(0, min(100, inclusivity_base))

        exclusive_penalty = len(bias_analysis.exclusive_words) * self.weights['exclusive_penalty']
        openness_base = 60 + exclusive_penalty
        scores['openness'] = max(0, min(100, openness_base))

        word_count = text_stats['word_count']
        if 50 <= word_count <= 300:
            length_score = 80
        elif 300 < word_count <= 500:
            length_score = 70
        elif word_count < 50:
            length_score = 30
        else:
            length_score = 60

        scores['text_quality'] = length_score

        sentiment_base = 50 + (sentiment_score * 50)
        scores['sentiment'] = max(0, min(100, sentiment_base))

        return scores

    def calculate_overall_score(self, component_scores: Dict[str, float]) -> float:
        weights = {
            'language_balance': 0.40,
            'inclusivity': 0.25,
            'openness': 0.20,
            'text_quality': 0.10,
            'sentiment': 0.05
        }

        overall = sum(
            component_scores[component] * weight
            for component, weight in weights.items()
            if component in component_scores
        )

        return round(overall, 1)

    def analyze_strengths_weaknesses(self, component_scores: Dict[str, float]) -> Tuple[List[str], List[str]]:
        strengths = []
        weaknesses = []

        # 定义评分标准
        thresholds = {
            'excellent': 80,
            'good': 65,
            'fair': 50,
            'poor': 35
        }

        for component, score in component_scores.items():
            if score >= thresholds['excellent']:
                if component == 'language_balance':
                    strengths.append("Language expression is balanced, avoiding gender bias")
                elif component == 'inclusivity':
                    strengths.append("Rich diversity and inclusivity vocabulary")
                elif component == 'openness':
                    strengths.append("Job requirements are open and flexible")
                elif component == 'text_quality':
                    strengths.append("Clear text structure with complete information")
                elif component == 'sentiment':
                    strengths.append("Overall positive and welcoming tone")

            elif score < thresholds['fair']:
                if component == 'language_balance':
                    weaknesses.append("Noticeable gender bias in language")
                elif component == 'inclusivity':
                    weaknesses.append("Lacks inclusive and diversity-related vocabulary")
                elif component == 'openness':
                    weaknesses.append("Job requirements are too strict or exclusive")
                elif component == 'text_quality':
                    weaknesses.append("Text length or structure needs optimization")
                elif component == 'sentiment':
                    weaknesses.append("Overall tone is somewhat negative")

        return strengths, weaknesses

    def generate_recommendations(self, component_scores: Dict[str, float], bias_analysis) -> List[str]:
        recommendations = []

        if component_scores.get('language_balance', 100) < 60:
            if len(bias_analysis.masculine_words) > 0:
                recommendations.append(
                    f"Replace masculine words: {', '.join(bias_analysis.masculine_words[:3])} "
                    "→ Use more neutral expressions"
                )
            recommendations.append("Add collaborative words like 'teamwork', 'communication', 'support'")

        if component_scores.get('inclusivity', 100) < 50:
            recommendations.append(
                "Add inclusive expressions: 'diverse team', 'equal opportunity', 'career development'")
            recommendations.append("Emphasize work-life balance and flexibility")

        if component_scores.get('openness', 100) < 50:
            if len(bias_analysis.exclusive_words) > 0:
                recommendations.append(
                    f"Soften strict requirements: change '{bias_analysis.exclusive_words[0]}' to 'preferred' or 'desired'"
                )
            recommendations.append("Use welcoming language like 'welcome', 'encourage', 'invite'")

        if component_scores.get('text_quality', 100) < 50:
            text_len = sum([len(bias_analysis.masculine_words), len(bias_analysis.feminine_words)])
            if text_len < 10:
                recommendations.append("Add more detailed and appealing job description content")
            else:
                recommendations.append("Simplify and streamline job description, highlight core requirements")

        if not recommendations:
            recommendations.append("Job description performs well overall, continue maintaining inclusive expression")

        return recommendations

    def score_job_description(self, text: str) -> InclusivityScore:
        component_scores = self.calculate_component_scores(text)

        overall_score = self.calculate_overall_score(component_scores)

        grade = get_score_label(overall_score)
        color = get_score_color(overall_score)

        bias_analysis = self.bias_detector.analyze_bias_patterns(text)

        strengths, weaknesses = self.analyze_strengths_weaknesses(component_scores)

        recommendations = self.generate_recommendations(component_scores, bias_analysis)

        return InclusivityScore(
            overall_score=overall_score,
            component_scores=component_scores,
            grade=grade,
            color=color,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations
        )

    def compare_scores(self, texts: List[str]) -> Dict:
        results = []
        for i, text in enumerate(texts):
            score_result = self.score_job_description(text)
            results.append({
                'index': i,
                'score': score_result.overall_score,
                'grade': score_result.grade,
                'result': score_result
            })

        results.sort(key=lambda x: x['score'], reverse=True)

        return {
            'results': results,
            'best_score': results[0]['score'] if results else 0,
            'worst_score': results[-1]['score'] if results else 0,
            'average_score': sum(r['score'] for r in results) / len(results) if results else 0
        }

    def get_score_distribution_stats(self, scores: List[float]) -> Dict:
        if not scores:
            return {}

        return {
            'count': len(scores),
            'mean': sum(scores) / len(scores),
            'min': min(scores),
            'max': max(scores),
            'std': math.sqrt(sum((x - sum(scores) / len(scores)) ** 2 for x in scores) / len(scores)),
            'distribution': {
                'excellent': len([s for s in scores if s >= 80]),
                'good': len([s for s in scores if 65 <= s < 80]),
                'fair': len([s for s in scores if 50 <= s < 65]),
                'poor': len([s for s in scores if s < 50])
            }
        }


inclusivity_scorer = InclusivityScorer()


def get_inclusivity_scorer() -> InclusivityScorer:
    return inclusivity_scorer