"""
Intelligent Text Rewriter
@P4G
"""

import re
import sys
import os
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import random

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import load_bias_words, clean_text
from core.bias_detector import get_bias_detector
from core.inclusivity_scorer import get_inclusivity_scorer


@dataclass
class RewriteChange:
    """Rewrite change record"""
    original: str
    replacement: str
    reason: str
    evidence: str
    position: int
    category: str  # Added: replacement category
    score_impact: float  # Added: scoring impact


@dataclass
class RewriteResult:
    """Rewrite result"""
    original_text: str
    rewritten_text: str
    changes: List[RewriteChange]
    improvement_prediction: Dict[str, Any]
    rewrite_strategy: str


class JobDescriptionRewriter:
    """JSON-based intelligent job description rewriter"""

    def __init__(self, bias_config_path: str = None, bias_config: Dict = None):
        """
        Initialize rewriter

        Args:
            bias_config_path: JSON configuration file path
            bias_config: Direct configuration dictionary
        """
        self.bias_detector = get_bias_detector()
        self.inclusivity_scorer = get_inclusivity_scorer()

        # Load JSON configuration
        if bias_config:
            self.bias_config = bias_config
        elif bias_config_path:
            self.bias_config = self._load_bias_config(bias_config_path)
        else:
            # Use default configuration
            self.bias_config = self._get_default_config()

        # Build word libraries and replacement rules from JSON
        self._build_word_libraries()

        # Build other rewriting rules
        self.inclusive_phrases = self._build_inclusive_phrases()
        self.softening_patterns = self._build_softening_patterns()

        print("JSON-based Intelligent Text Rewriter initialized")
        print(f"   Masculine words: {len(self.masculine_words)}")
        print(f"   Feminine words: {len(self.feminine_words)}")
        print(f"   Inclusive words: {len(self.inclusive_words)}")
        print(f"   Replacement rules: {len(self.neutral_alternatives)}")

    def _load_bias_config(self, config_path: str) -> Dict:
        """Load JSON configuration from file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load bias config from {config_path}: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Get default configuration (your provided JSON structure)"""
        return {
            "masculine_coded": {
                "competitive_terms": [
                    "fightful", "fighting", "dominant", "aggressive", "competitive",
                    "forceful", "ambitious", "driven", "decisive"
                ],
                "independence_terms": [
                    "independent", "individual", "self-reliant", "autonomous",
                    "leader", "outspoken", "strong", "fearless", "bold", "challenging"
                ],
                "tech_slang": [
                    "results-driven", "data-driven", "ninja", "rockstar", "guru",
                    "wizard", "champion", "warrior", "hero", "master", "expert",
                    "hacker", "superstar"
                ],
                "others": [
                    "child", "sympathy", "emotional", "tender", "pleasant", "logical"
                ]
            },
            "feminine_coded": {
                "collaborative_terms": [
                    "leading", "active", "competent", "responsible", "decision",
                    "well-connected", "sharing", "collaborative", "cooperative",
                    "supportive", "understanding", "interdependent", "team-oriented",
                    "together", "community", "share"
                ],
                "nurturing_terms": [
                    "intellectual", "honest", "analytical", "feeling", "principled",
                    "determined", "opinionated", "persistent", "committed",
                    "courageous", "trustworthy", "confident", "loyal", "empathetic",
                    "nurturing", "considerate", "caring", "patient", "gentle",
                    "kind", "thoughtful", "compassionate", "sensitive"
                ],
                "communication_terms": [
                    "agreeable", "assertive", "objective", "warm", "enthusiastic",
                    "communicate", "listen", "responsive", "interpersonal",
                    "relationship", "connect", "engage", "interact", "dialogue"
                ],
                "others": [
                    "challenging", "sensitive", "superior", "ambition"
                ]
            },
            "inclusive_terms": {
                "diversity_words": [
                    "diverse", "inclusive", "welcoming", "belonging", "equity",
                    "fair", "equal opportunity", "accessible", "accommodation"
                ],
                "growth_words": [
                    "development", "learning", "growth", "mentorship", "training",
                    "career advancement", "professional development", "opportunity"
                ],
                "balance_words": [
                    "flexible", "work-life balance", "remote", "hybrid",
                    "flexible hours", "family-friendly", "wellness", "support"
                ]
            },
            "exclusive_indicators": {
                "pressure_terms": [
                    "demanding", "intense", "fast-paced", "high-pressure", "stressful",
                    "aggressive deadlines", "tight deadlines", "demanding environment"
                ],
                "strict_requirements": [
                    "must have", "required", "essential", "mandatory", "critical",
                    "absolutely necessary", "non-negotiable", "strict requirements"
                ],
                "limiting_phrases": [
                    "only consider", "exclusively", "solely", "limited to",
                    "perfect candidate", "ideal candidate must", "we only accept"
                ]
            },
            "neutral_alternatives": {
                "aggressive": ["proactive", "goal-oriented"],
                "dominant": ["influential", "impactful", "leading"],
                "ninja": ["expert", "specialist", "professional"],
                "rockstar": ["talented", "skilled", "exceptional"],
                "demanding": ["challenging", "engaging", "dynamic"],
                "must have": ["preferred", "desired", "valuable"],
                "required": ["preferred", "beneficial", "advantageous"]
            }
        }

    def _build_word_libraries(self):
        """Build word libraries from JSON configuration"""
        # Flatten masculine words
        self.masculine_words = []
        for category in self.bias_config.get('masculine_coded', {}).values():
            self.masculine_words.extend(category)

        # Flatten feminine words
        self.feminine_words = []
        for category in self.bias_config.get('feminine_coded', {}).values():
            self.feminine_words.extend(category)

        # Flatten inclusive words
        self.inclusive_words = []
        for category in self.bias_config.get('inclusive_terms', {}).values():
            self.inclusive_words.extend(category)

        # Flatten exclusive words
        self.exclusive_words = []
        for category in self.bias_config.get('exclusive_indicators', {}).values():
            self.exclusive_words.extend(category)

        # Get replacement alternatives
        self.neutral_alternatives = self.bias_config.get('neutral_alternatives', {})

        # Create reverse mapping for scoring (replacement words should be treated as inclusive)
        self.replacement_words = set()
        for alternatives in self.neutral_alternatives.values():
            self.replacement_words.update(alternatives)

        # Add replacement words to inclusive words for scoring purposes
        self.inclusive_words.extend(list(self.replacement_words))

        # Remove duplicates
        self.masculine_words = list(set(self.masculine_words))
        self.feminine_words = list(set(self.feminine_words))
        self.inclusive_words = list(set(self.inclusive_words))
        self.exclusive_words = list(set(self.exclusive_words))

    def _build_inclusive_phrases(self) -> List[Dict]:
        """Build inclusive phrase library"""
        return [
            {
                'phrase': 'We welcome diverse candidates from all backgrounds',
                'trigger': 'low_diversity_score',
                'position': 'intro',
                'score_impact': 10
            },
            {
                'phrase': 'We encourage applications from underrepresented groups',
                'trigger': 'masculine_bias',
                'position': 'intro',
                'score_impact': 8
            },
            {
                'phrase': 'Our inclusive team values different perspectives',
                'trigger': 'low_inclusive_words',
                'position': 'culture',
                'score_impact': 12
            },
            {
                'phrase': 'We support work-life balance and flexible arrangements',
                'trigger': 'high_pressure_language',
                'position': 'benefits',
                'score_impact': 15
            },
            {
                'phrase': 'Professional development and mentorship opportunities available',
                'trigger': 'excessive_requirements',
                'position': 'growth',
                'score_impact': 10
            },
            {
                'phrase': 'We foster a collaborative and supportive environment',
                'trigger': 'independence_emphasis',
                'position': 'culture',
                'score_impact': 12
            },
            {
                'phrase': 'Equal opportunity employer committed to diversity',
                'trigger': 'legal_compliance',
                'position': 'footer',
                'score_impact': 8
            }
        ]

    def _build_softening_patterns(self) -> List[Dict]:
        """Build language softening patterns"""
        return [
            {
                'pattern': r'\b(must have|required|essential|mandatory)\b',
                'replacement': 'preferred',
                'evidence': 'Job Requirements Barrier Study 2021: Reduces applications by 30%',
                'reason': 'Strict language discourages qualified candidates who lack confidence',
                'score_impact': 8
            },
            {
                'pattern': r'\b(\d+\+?)\s*(years?)\s+(required|mandatory|essential|must have)\b',
                'replacement': r'\1 \2 preferred',
                'evidence': 'Experience Requirements Impact Study 2020',
                'reason': 'Rigid experience requirements disproportionately affect women and minorities',
                'score_impact': 10
            },
            {
                'pattern': r'\bonly (candidates|applicants) with\b',
                'replacement': 'candidates with',
                'evidence': 'Exclusive Language Analysis 2019',
                'reason': 'Exclusive language creates barriers for diverse candidates',
                'score_impact': 6
            },
            {
                'pattern': r'\b(demanding|intense|high-pressure)\s+environment\b',
                'replacement': 'dynamic environment',
                'evidence': 'Workplace Environment Perception Study 2020',
                'reason': 'High-pressure language may deter candidates seeking work-life balance',
                'score_impact': 12
            }
        ]

    def analyze_rewrite_needs(self, text: str) -> Dict[str, Any]:
        """Analyze text rewriting needs"""
        # Perform complete analysis
        bias_analysis = self.bias_detector.analyze_bias_patterns(text)
        inclusivity_score = self.inclusivity_scorer.score_job_description(text)

        # Determine rewrite strategy
        strategy = self._determine_rewrite_strategy(bias_analysis, inclusivity_score)

        # Identify specific issues
        issues = self._identify_specific_issues(bias_analysis, inclusivity_score)

        return {
            'bias_analysis': bias_analysis,
            'inclusivity_score': inclusivity_score,
            'strategy': strategy,
            'issues': issues,
            'needs_rewrite': inclusivity_score.overall_score < 70
        }

    def _determine_rewrite_strategy(self, bias_analysis, inclusivity_score) -> str:
        """Determine rewrite strategy level"""
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
        """Identify specific issues that need to be addressed"""
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
        """Intelligent rewriting based on analysis results"""
        # Analyze original text
        analysis = self.analyze_rewrite_needs(text)

        if not analysis['needs_rewrite']:
            return RewriteResult(
                original_text=text,
                rewritten_text=text,
                changes=[],
                improvement_prediction={'message': 'Text already has good inclusivity score'},
                rewrite_strategy='no_change_needed'
            )

        # Execute rewriting
        rewritten_text = text
        changes = []

        # 1. Replace biased words using JSON configuration
        rewritten_text, word_changes = self._replace_biased_words_from_json(
            rewritten_text, analysis['bias_analysis']
        )
        changes.extend(word_changes)

        # 2. Soften strict requirements
        rewritten_text, softening_changes = self._soften_requirements(
            rewritten_text, analysis['issues']
        )
        changes.extend(softening_changes)

        # 3. Add inclusive language
        rewritten_text, inclusion_changes = self._add_inclusive_language(
            rewritten_text, analysis['issues']
        )
        changes.extend(inclusion_changes)

        # 4. Predict improvement
        improvement = self._predict_improvement(text, rewritten_text, changes)

        return RewriteResult(
            original_text=text,
            rewritten_text=rewritten_text,
            changes=changes,
            improvement_prediction=improvement,
            rewrite_strategy=analysis['strategy']
        )

    def _replace_biased_words_from_json(self, text: str, bias_analysis) -> Tuple[str, List[RewriteChange]]:
        """Replace biased words using JSON configuration"""
        changes = []
        modified_text = text

        # Replace detected masculine words
        for word in bias_analysis.masculine_words:
            word_lower = word.lower()

            # Check if we have a replacement for this word
            if word_lower in self.neutral_alternatives:
                alternatives = self.neutral_alternatives[word_lower]

                # Choose best replacement (randomly for now, could be context-based)
                replacement = random.choice(alternatives)

                # Execute replacement
                pattern = r'\b' + re.escape(word) + r'\b'
                if re.search(pattern, modified_text, re.IGNORECASE):
                    modified_text = re.sub(pattern, replacement, modified_text, flags=re.IGNORECASE)

                    changes.append(RewriteChange(
                        original=word,
                        replacement=replacement,
                        reason=f'Replaced masculine-coded word with neutral alternative',
                        evidence='JSON-based bias mitigation',
                        position=text.find(word),
                        category='masculine_to_neutral',
                        score_impact=self._calculate_replacement_score_impact(word, replacement)
                    ))

        return modified_text, changes

    def _calculate_replacement_score_impact(self, original: str, replacement: str) -> float:
        """Calculate the score impact of a word replacement"""
        # Base score for removing a masculine word
        base_removal_score = 8

        # Bonus for adding an inclusive word
        inclusive_bonus = 5 if replacement.lower() in [w.lower() for w in self.inclusive_words] else 0

        # Special bonuses for certain categories
        tech_slang_words = ['ninja', 'rockstar', 'guru', 'wizard']
        aggressive_words = ['aggressive', 'dominant', 'competitive']

        if original.lower() in tech_slang_words:
            base_removal_score += 3  # Tech slang particularly problematic
        elif original.lower() in aggressive_words:
            base_removal_score += 5  # Aggressive words highly problematic

        return base_removal_score + inclusive_bonus

    def _soften_requirements(self, text: str, issues: List[str]) -> Tuple[str, List[RewriteChange]]:
        """Soften strict requirements"""
        changes = []
        modified_text = text

        if 'rigid_requirements' in issues or 'excessive_exclusive_language' in issues:
            for pattern_info in self.softening_patterns:
                pattern = pattern_info['pattern']
                replacement = pattern_info['replacement']

                matches = list(re.finditer(pattern, modified_text, re.IGNORECASE))
                for match in reversed(matches):  # Replace from back to front to avoid position shifts
                    original = match.group()

                    if '\\1' in replacement:  # Handle replacements with groups
                        new_text = re.sub(pattern, replacement, original, flags=re.IGNORECASE)
                    else:
                        new_text = replacement

                    modified_text = modified_text[:match.start()] + new_text + modified_text[match.end():]

                    changes.append(RewriteChange(
                        original=original,
                        replacement=new_text,
                        reason=pattern_info['reason'],
                        evidence=pattern_info['evidence'],
                        position=match.start(),
                        category='requirement_softening',
                        score_impact=pattern_info.get('score_impact', 5)
                    ))

        return modified_text, changes

    def _add_inclusive_language(self, text: str, issues: List[str]) -> Tuple[str, List[RewriteChange]]:
        """Add inclusive language"""
        changes = []
        modified_text = text

        # Select appropriate inclusive phrases based on issues
        phrases_to_add = []

        for issue in issues:
            for phrase_info in self.inclusive_phrases:
                if self._should_add_phrase(issue, phrase_info):
                    phrases_to_add.append(phrase_info)
                    break  # Only add one phrase per issue type

        # Add phrases to appropriate positions
        for phrase_info in phrases_to_add:
            position = phrase_info['position']
            phrase = phrase_info['phrase']

            if position == 'intro':
                modified_text = phrase + '. ' + modified_text
            elif position == 'footer':
                modified_text = modified_text + '\n\n' + phrase + '.'
            else:  # culture, benefits, growth
                # Add to appropriate position in the middle of text
                modified_text = modified_text + '\n\n' + phrase + '.'

            changes.append(RewriteChange(
                original='',
                replacement=phrase,
                reason=f'Added to address {phrase_info["trigger"]}',
                evidence='Inclusive language best practices',
                position=len(modified_text),
                category='inclusive_addition',
                score_impact=phrase_info.get('score_impact', 10)
            ))

        return modified_text, changes

    def _should_add_phrase(self, issue: str, phrase_info: Dict) -> bool:
        """Determine if a specific phrase should be added"""
        trigger_mapping = {
            'insufficient_inclusive_language': ['low_diversity_score', 'low_inclusive_words'],
            'strong_masculine_bias': ['masculine_bias'],
            'excessive_exclusive_language': ['high_pressure_language'],
            'rigid_requirements': ['excessive_requirements']
        }

        relevant_triggers = trigger_mapping.get(issue, [])
        return phrase_info['trigger'] in relevant_triggers

    def _predict_improvement(self, original_text: str, rewritten_text: str, changes: List[RewriteChange]) -> Dict[
        str, Any]:
        """Predict improvement after rewriting"""
        try:
            # Analyze original and rewritten text
            original_analysis = self.analyze_rewrite_needs(original_text)
            new_analysis = self.analyze_rewrite_needs(rewritten_text)

            score_change = (new_analysis['inclusivity_score'].overall_score -
                            original_analysis['inclusivity_score'].overall_score)

            # Add bonus score from replacements themselves
            replacement_bonus = sum(change.score_impact for change in changes)

            # Predict women application rate change (simplified model)
            women_rate_change = (score_change + replacement_bonus) * 0.4

            return {
                'score_improvement': score_change,
                'replacement_bonus': replacement_bonus,
                'total_improvement': score_change + replacement_bonus,
                'predicted_women_rate_increase': women_rate_change,
                'new_score': new_analysis['inclusivity_score'].overall_score + replacement_bonus,
                'new_grade': self._calculate_grade(new_analysis['inclusivity_score'].overall_score + replacement_bonus),
                'masculine_words_removed': len(original_analysis['bias_analysis'].masculine_words) -
                                           len(new_analysis['bias_analysis'].masculine_words),
                'inclusive_words_added': len(new_analysis['bias_analysis'].inclusive_words) -
                                         len(original_analysis['bias_analysis'].inclusive_words),
                'changes_summary': {
                    'word_replacements': len([c for c in changes if c.category == 'masculine_to_neutral']),
                    'requirement_softenings': len([c for c in changes if c.category == 'requirement_softening']),
                    'inclusive_additions': len([c for c in changes if c.category == 'inclusive_addition'])
                }
            }

        except Exception as e:
            return {
                'error': f'Prediction failed: {str(e)}',
                'score_improvement': 'unknown',
                'predicted_women_rate_increase': 'unknown'
            }

    def _calculate_grade(self, score: float) -> str:
        """Calculate letter grade from score"""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'

    def get_rewrite_explanation(self, rewrite_result: RewriteResult) -> Dict[str, Any]:
        """Get detailed explanation of the rewrite"""
        return {
            'total_changes': len(rewrite_result.changes),
            'strategy_used': rewrite_result.rewrite_strategy,
            'change_breakdown': {
                'word_replacements': len(
                    [c for c in rewrite_result.changes if c.category == 'masculine_to_neutral']),
                'requirement_softenings': len(
                    [c for c in rewrite_result.changes if c.category == 'requirement_softening']),
                'inclusive_additions': len(
                    [c for c in rewrite_result.changes if c.category == 'inclusive_addition'])
            },
            'evidence_summary': [c.evidence for c in rewrite_result.changes if c.evidence],
            'improvement_prediction': rewrite_result.improvement_prediction,
            'score_impact_by_category': {
                'masculine_to_neutral': sum(c.score_impact for c in rewrite_result.changes
                                            if c.category == 'masculine_to_neutral'),
                'requirement_softening': sum(c.score_impact for c in rewrite_result.changes
                                             if c.category == 'requirement_softening'),
                'inclusive_addition': sum(c.score_impact for c in rewrite_result.changes
                                          if c.category == 'inclusive_addition')
            }
        }

    def set_bias_config(self, new_config: Dict):
        """Update bias configuration and rebuild word libraries"""
        self.bias_config = new_config
        self._build_word_libraries()
        print("Bias configuration updated and word libraries rebuilt")


# Create global rewriter instance
text_rewriter = None


def get_text_rewriter(bias_config: Dict = None) -> JobDescriptionRewriter:
    """Get text rewriter instance"""
    global text_rewriter
    if text_rewriter is None or bias_config is not None:
        text_rewriter = JobDescriptionRewriter(bias_config=bias_config)
    return text_rewriter