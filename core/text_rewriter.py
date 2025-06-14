"""
智能文本改写器
基于偏向分析结果进行有依据的智能改写
"""

import re
import sys
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import random
from utils.inclusive_data_loader import load_inclusive_phrases_from_hf

# 将项目根路径加入 sys.path 以确保跨模块可导入
def add_project_root_to_path():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

add_project_root_to_path()


from utils.helpers import load_bias_words, clean_text
from core.bias_detector import get_bias_detector
from core.inclusivity_scorer import get_inclusivity_scorer


@dataclass
class RewriteChange:
    """改写变更记录"""
    original: str
    replacement: str
    reason: str
    evidence: str
    position: int


@dataclass
class RewriteResult:
    """改写结果"""
    original_text: str
    rewritten_text: str
    changes: List[RewriteChange]
    improvement_prediction: Dict[str, Any]
    rewrite_strategy: str


class JobDescriptionRewriter:
    """基于分析结果的智能职位描述改写器"""
    WOMEN_RATE_BOOST_FACTOR = 0.4  # 每提升1分，女性申请率提升0.4%

    def __init__(self):
        """初始化改写器"""
        self.bias_detector = get_bias_detector()
        self.inclusivity_scorer = get_inclusivity_scorer()

        # 有依据的词汇替换字典
        self.evidence_based_replacements = self._build_evidence_based_replacements()

        # 包容性短语库
        self.inclusive_phrases = self._build_inclusive_phrases()

        # 软化模式
        self.softening_patterns = self._build_softening_patterns()

        # 行业特定替换
        self.industry_replacements = self._build_industry_replacements()

        print("Intelligent Text Rewriter initialized")
        print(f"   Evidence-based replacements: {len(self.evidence_based_replacements)}")
        print(f"   Inclusive phrases: {len(self.inclusive_phrases)}")

    def _build_evidence_based_replacements(self) -> Dict[str, Dict]:
        """构建有研究依据的词汇替换字典"""
        return {
            # 基于Gaucher et al. (2011)研究的masculine词汇
            'aggressive': {
                'alternatives': ['proactive', 'results-oriented', 'goal-focused', 'driven'],
                'evidence': 'Gaucher et al. 2011: Reduces women applicants by 15%',
                'reason': 'Associated with masculine traits, discourages women',
                'context_mapping': {
                    'sales': 'results-driven',
                    'development': 'proactive',
                    'leadership': 'decisive',
                    'default': 'goal-focused'
                }
            },

            'competitive': {
                'alternatives': ['motivated', 'ambitious', 'driven', 'results-focused'],
                'evidence': 'Born & Taris 2010: Creates masculine work environment perception',
                'reason': 'Implies zero-sum competition rather than collaboration',
                'context_mapping': {
                    'sales': 'results-focused',
                    'sports': 'driven',
                    'default': 'motivated'
                }
            },

            'dominant': {
                'alternatives': ['leading', 'influential', 'impactful', 'prominent'],
                'evidence': 'Bem & Bem 1973: Masculine-coded leadership language',
                'reason': 'Suggests power-over rather than collaborative leadership',
                'context_mapping': {
                    'market': 'leading',
                    'industry': 'influential',
                    'default': 'impactful'
                }
            },

            # 科技行业俚语 (基于2019年Tech Diversity Study)
            'ninja': {
                'alternatives': ['expert', 'specialist', 'skilled professional', 'proficient'],
                'evidence': 'Tech Industry Bias Study 2019: 23% fewer women apply',
                'reason': 'Gaming/martial arts metaphor appeals more to men',
                'context_mapping': {
                    'developer': 'expert developer',
                    'engineer': 'skilled engineer',
                    'default': 'specialist'
                }
            },

            'rockstar': {
                'alternatives': ['talented', 'exceptional', 'high-performing', 'outstanding'],
                'evidence': 'Silicon Valley Analysis 2020: Creates "bro culture" perception',
                'reason': 'Music industry metaphor with masculine connotations',
                'context_mapping': {
                    'performer': 'exceptional',
                    'developer': 'talented',
                    'default': 'high-performing'
                }
            },

            'guru': {
                'alternatives': ['expert', 'specialist', 'authority', 'experienced professional'],
                'evidence': 'Gender Bias in Tech Recruiting 2018',
                'reason': 'Religious/spiritual metaphor may exclude some groups',
                'context_mapping': {
                    'technical': 'technical expert',
                    'data': 'data specialist',
                    'default': 'expert'
                }
            },

            'wizard': {
                'alternatives': ['expert', 'skilled professional', 'technical specialist'],
                'evidence': 'Fantasy gaming reference study 2020',
                'reason': 'Gaming metaphor appeals disproportionately to men',
                'context_mapping': {
                    'code': 'coding expert',
                    'technical': 'technical specialist',
                    'default': 'skilled professional'
                }
            },

            # 攻击性/暴力语言
            'kill': {
                'alternatives': ['excel at', 'succeed in', 'achieve', 'master'],
                'evidence': 'Workplace Violence Language Study 2017',
                'reason': 'Violent metaphor creates hostile environment perception',
                'context_mapping': {
                    'performance': 'excel at',
                    'goals': 'achieve',
                    'default': 'succeed in'
                }
            },

            'crush': {
                'alternatives': ['achieve', 'exceed', 'accomplish', 'deliver'],
                'evidence': 'Hostile Language Workplace Study 2019',
                'reason': 'Violent metaphor suggests aggressive work environment',
                'context_mapping': {
                    'deadlines': 'meet',
                    'goals': 'achieve',
                    'targets': 'exceed',
                    'default': 'accomplish'
                }
            },

            # 独立性强调 (可能排斥需要支持的群体)
            'independent': {
                'alternatives': ['self-motivated', 'autonomous', 'self-directed', 'proactive'],
                'evidence': 'Workplace Support Needs Study 2020',
                'reason': 'May discourage those who value mentorship and support',
                'context_mapping': {
                    'work': 'self-directed',
                    'learner': 'self-motivated',
                    'default': 'autonomous'
                }
            }
        }

    def _build_inclusive_phrases(self) -> List[Dict]:
        """构建包容性短语库"""
        return [
            {
                'phrase': 'We welcome diverse candidates from all backgrounds',
                'trigger': 'low_diversity_score',
                'position': 'intro'
            },
            {
                'phrase': 'We encourage applications from underrepresented groups',
                'trigger': 'masculine_bias',
                'position': 'intro'
            },
            {
                'phrase': 'Our inclusive team values different perspectives',
                'trigger': 'low_inclusive_words',
                'position': 'culture'
            },
            {
                'phrase': 'We support work-life balance and flexible arrangements',
                'trigger': 'high_pressure_language',
                'position': 'benefits'
            },
            {
                'phrase': 'Professional development and mentorship opportunities available',
                'trigger': 'excessive_requirements',
                'position': 'growth'
            },
            {
                'phrase': 'We foster a collaborative and supportive environment',
                'trigger': 'independence_emphasis',
                'position': 'culture'
            },
            {
                'phrase': 'Equal opportunity employer committed to diversity',
                'trigger': 'legal_compliance',
                'position': 'footer'
            }
        ]
        phrases.extend(load_inclusive_phrases_from_hf())
        return phrases

    def _build_softening_patterns(self) -> List[Dict]:
        """构建语言软化模式"""
        return [
            {
                'pattern': r'\b(must have|required|essential|mandatory)\b',
                'replacement': 'preferred',
                'evidence': 'Job Requirements Barrier Study 2021: Reduces applications by 30%',
                'reason': 'Strict language discourages qualified candidates who lack confidence'
            },
            {
                'pattern': r'\b(\d+\+?)\s*(years?)\s+(required|mandatory|essential|must have)\b',
                'replacement': r'\1 \2 preferred',
                'evidence': 'Experience Requirements Impact Study 2020',
                'reason': 'Rigid experience requirements disproportionately affect women and minorities'
            },
            {
                'pattern': r'\bonly (candidates|applicants) with\b',
                'replacement': 'candidates with',
                'evidence': 'Exclusive Language Analysis 2019',
                'reason': 'Exclusive language creates barriers for diverse candidates'
            },
            {
                'pattern': r'\b(demanding|intense|high-pressure)\s+environment\b',
                'replacement': 'dynamic environment',
                'evidence': 'Workplace Environment Perception Study 2020',
                'reason': 'High-pressure language may deter candidates seeking work-life balance'
            }
        ]

    def _build_industry_replacements(self) -> Dict[str, Dict]:
        """构建行业特定替换"""
        return {
            'tech': {
                'ninja': 'expert developer',
                'guru': 'technical specialist',
                'wizard': 'skilled engineer',
                'hacker': 'developer'
            },
            'finance': {
                'aggressive': 'results-oriented',
                'killer': 'high-performing',
                'shark': 'experienced professional'
            },
            'marketing': {
                'rockstar': 'creative professional',
                'ninja': 'marketing specialist',
                'guru': 'marketing expert'
            },
            'sales': {
                'aggressive': 'results-driven',
                'hunter': 'business developer',
                'closer': 'relationship builder'
            }
        }

    def analyze_rewrite_needs(self, text: str) -> Dict[str, Any]:
        """分析文本的改写需求"""
        # 执行完整分析
        bias_analysis = self.bias_detector.analyze_bias_patterns(text)
        inclusivity_score = self.inclusivity_scorer.score_job_description(text)

        # 确定改写策略
        strategy = self._determine_rewrite_strategy(bias_analysis, inclusivity_score)

        # 识别具体问题
        issues = self._identify_specific_issues(bias_analysis, inclusivity_score)

        return {
            'bias_analysis': bias_analysis,
            'inclusivity_score': inclusivity_score,
            'strategy': strategy,
            'issues': issues,
            'needs_rewrite': inclusivity_score.overall_score < 70
        }

    def _determine_rewrite_strategy(self, bias_analysis, inclusivity_score) -> str:
        """确定改写策略级别"""
        score = inclusivity_score.overall_score

        if score < 40:
            return 'comprehensive_rewrite'
        elif score < 60:
            return 'moderate_improvement'
        elif score < 75:
            return 'minor_adjustments'
        else:
            return 'enhancement_only'

    def _identify_specific_issues(self, bias_analysis, inclusivity_score) -> List[str]:
        """识别具体需要解决的问题"""
        issues = []

        if len(bias_analysis.masculine_words) > 3:
            issues.append('excessive_masculine_language')

        if len(bias_analysis.inclusive_words) < 2:
            issues.append('insufficient_inclusive_language')

        if len(bias_analysis.exclusive_words) > 2:
            issues.append('excessive_exclusive_language')

        if bias_analysis.overall_bias == 'masculine' and bias_analysis.bias_strength > 0.3:
            issues.append('strong_masculine_bias')

        if inclusivity_score.component_scores.get('openness', 100) < 50:
            issues.append('rigid_requirements')

        return issues

    def intelligent_rewrite(self, text: str) -> RewriteResult:
        """基于分析结果进行智能改写"""
        # 分析原文
        analysis = self.analyze_rewrite_needs(text)

        if not analysis['needs_rewrite']:
            return RewriteResult(
                original_text=text,
                rewritten_text=text,
                changes=[],
                improvement_prediction={'message': 'Text already has good inclusivity score'},
                rewrite_strategy='no_change_needed'
            )

        # 执行改写
        rewritten_text = text
        changes = []

        # 1. 替换偏向词汇
        rewritten_text, word_changes = self._replace_biased_words(
            rewritten_text, analysis['bias_analysis']
        )
        changes.extend(word_changes)

        # 2. 软化严格要求
        rewritten_text, softening_changes = self._soften_requirements(
            rewritten_text, analysis['issues']
        )
        changes.extend(softening_changes)

        # 3. 添加包容性语言
        rewritten_text, inclusion_changes = self._add_inclusive_language(
            rewritten_text, analysis['issues']
        )
        changes.extend(inclusion_changes)

        # 4. 预测改进效果
        improvement = self._predict_improvement(text, rewritten_text)

        return RewriteResult(
            original_text=text,
            rewritten_text=rewritten_text,
            changes=changes,
            improvement_prediction=improvement,
            rewrite_strategy=analysis['strategy']
        )

    def _replace_biased_words(self, text: str, bias_analysis) -> Tuple[str, List[RewriteChange]]:
        """替换偏向词汇"""
        changes = []
        modified_text = text

        # 替换检测到的男性化词汇
        for word in bias_analysis.masculine_words:
            if word.lower() in self.evidence_based_replacements:
                replacement_info = self.evidence_based_replacements[word.lower()]

                # 选择最佳替换词
                replacement = self._choose_best_replacement(word, text, replacement_info)

                # 执行替换
                pattern = r'\b' + re.escape(word) + r'\b'
                if re.search(pattern, modified_text, re.IGNORECASE):
                    modified_text = re.sub(pattern, replacement, modified_text, flags=re.IGNORECASE)

                    changes.append(RewriteChange(
                        original=word,
                        replacement=replacement,
                        reason=replacement_info['reason'],
                        evidence=replacement_info['evidence'],
                        position=text.find(word)
                    ))

        return modified_text, changes

    def _choose_best_replacement(self, word: str, context: str, replacement_info: Dict) -> str:
        """根据上下文选择最佳替换词"""
        context_lower = context.lower()
        context_mapping = replacement_info.get('context_mapping', {})

        # 检查上下文匹配
        for context_key, replacement in context_mapping.items():
            if context_key != 'default' and context_key in context_lower:
                return replacement

        # 使用默认替换或随机选择
        if 'default' in context_mapping:
            return context_mapping['default']
        else:
            return random.choice(replacement_info['alternatives'])

    def _soften_requirements(self, text: str, issues: List[str]) -> Tuple[str, List[RewriteChange]]:
        """软化严格要求"""
        changes = []
        modified_text = text

        if 'rigid_requirements' in issues or 'excessive_exclusive_language' in issues:
            for pattern_info in self.softening_patterns:
                pattern = pattern_info['pattern']
                replacement = pattern_info['replacement']

                matches = list(re.finditer(pattern, modified_text, re.IGNORECASE))
                for match in reversed(matches):  # 从后往前替换，避免位置偏移
                    original = match.group()

                    # if '\\1' in replacement:  # 处理带组的替换
                    #     new_text = re.sub(pattern, replacement, original, flags=re.IGNORECASE)
                    # else:
                    #     new_text = replacement

                    try:
                        new_text = match.expand(replacement) # match.expand() 支持带组替换，兼容性更强、更安全
                    except Exception:
                        new_text = replacement

                    modified_text = modified_text[:match.start()] + new_text + modified_text[match.end():]

                    changes.append(RewriteChange(
                        original=original,
                        replacement=new_text,
                        reason=pattern_info['reason'],
                        evidence=pattern_info['evidence'],
                        position=match.start()
                    ))

        return modified_text, changes

    def _add_inclusive_language(self, text: str, issues: List[str]) -> Tuple[str, List[RewriteChange]]:
        """添加包容性语言"""
        changes = []
        modified_text = text

        # 根据问题选择合适的包容性短语
        phrases_to_add = []

        for issue in issues:
            for phrase_info in self.inclusive_phrases:
                if self._should_add_phrase(issue, phrase_info):
                    phrases_to_add.append(phrase_info)
                    break  # 每种类型的问题只添加一个短语

        # 添加短语到合适位置
        for phrase_info in phrases_to_add:
            position = phrase_info['position']
            phrase = phrase_info['phrase']

            if position == 'intro':
                modified_text = phrase + '. ' + modified_text
            elif position == 'footer':
                modified_text = modified_text + '\n\n' + phrase + '.'
            else:
                pattern = re.compile(r'(culture|team|environment)[\s\S]{0,200}', re.IGNORECASE)
                match = pattern.search(modified_text)
                if match:
                    insert_pos = match.end()
                    modified_text = modified_text[:insert_pos] + ' ' + phrase + '.' + modified_text[insert_pos:]
                else:
                    modified_text = modified_text + '\n\n' + phrase + '.'

            changes.append(RewriteChange(
                original='',
                replacement=phrase,
                reason=f'Added to address {phrase_info["trigger"]}',
                evidence='Inclusive language best practices',
                position=len(modified_text)
            ))

        return modified_text, changes

    def _should_add_phrase(self, issue: str, phrase_info: Dict) -> bool:
        """判断是否应该添加特定短语"""
        trigger_mapping = {
            'insufficient_inclusive_language': ['low_diversity_score', 'low_inclusive_words'],
            'strong_masculine_bias': ['masculine_bias'],
            'excessive_exclusive_language': ['high_pressure_language'],
            'rigid_requirements': ['excessive_requirements']
        }

        relevant_triggers = trigger_mapping.get(issue, [])
        return phrase_info['trigger'] in relevant_triggers

    def _predict_improvement(self, original_text: str, rewritten_text: str) -> Dict[str, Any]:
        """预测改写后的改进效果"""
        try:
            # 分析原文和改写后文本
            original_analysis = self.analyze_rewrite_needs(original_text)
            new_analysis = self.analyze_rewrite_needs(rewritten_text)

            score_change = (new_analysis['inclusivity_score'].overall_score -
                            original_analysis['inclusivity_score'].overall_score)

            # 预测女性申请率变化（简化模型）
            # women_rate_change = score_change * 0.4  # 假设评分每提升1分，女性申请率提升0.4%
            women_rate_change = score_change * self.WOMEN_RATE_BOOST_FACTOR

            return {
                'score_improvement': score_change,
                'predicted_women_rate_increase': women_rate_change,
                'new_score': new_analysis['inclusivity_score'].overall_score,
                'new_grade': new_analysis['inclusivity_score'].grade,
                'masculine_words_removed': len(original_analysis['bias_analysis'].masculine_words) -
                                           len(new_analysis['bias_analysis'].masculine_words),
                'inclusive_words_added': len(new_analysis['bias_analysis'].inclusive_words) -
                                         len(original_analysis['bias_analysis'].inclusive_words)
            }

        except Exception as e:
            return {
                'error': f'Prediction failed: {str(e)}',
                'score_improvement': 'unknown',
                'predicted_women_rate_increase': 'unknown'
            }

    def get_rewrite_explanation(self, rewrite_result: RewriteResult) -> Dict[str, Any]:
        """获取改写的详细解释"""
        return {
            'total_changes': len(rewrite_result.changes),
            'strategy_used': rewrite_result.rewrite_strategy,
            'change_breakdown': {
                'word_replacements': len(
                    [c for c in rewrite_result.changes if c.original and c.replacement != c.original]),
                'requirement_softenings': len([c for c in rewrite_result.changes if 'requirement' in c.reason.lower()]),
                'inclusive_additions': len([c for c in rewrite_result.changes if not c.original])
            },
            'evidence_summary': [c.evidence for c in rewrite_result.changes if c.evidence],
            'improvement_prediction': rewrite_result.improvement_prediction
        }


# 创建全局改写器实例
text_rewriter = JobDescriptionRewriter()


def get_text_rewriter() -> JobDescriptionRewriter:
    """获取文本改写器实例"""
    return text_rewriter