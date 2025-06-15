"""
GENIA@P4G
"""

import streamlit as st
import re
import json
from typing import Dict, List, Optional

# Set page config
st.set_page_config(
    page_title="GENIA@P4G",
    page_icon="🔍",
    layout="wide"
)

# Simple CSS
st.markdown("""
<style>
    .header-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }

    .metric-box {
        background: white;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        margin: 10px 0;
        text-align: center;
    }

    .highlight-masculine {
        background-color: #fca5a5;
        color: #991b1b;
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: bold;
    }

    .highlight-inclusive {
        background-color: #bfdbfe;
        color: #1e40af;
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: bold;
    }

    .highlight-exclusive {
        background-color: #fed7aa;
        color: #9a3412;
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: bold;
    }

    .highlight-replacement {
        background-color: #dcfce7;
        color: #166534;
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: bold;
    }

    .text-display {
        background: #f9fafb;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        line-height: 1.6;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Default JSON configuration
DEFAULT_BIAS_CONFIG = {
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
        "required": ["preferred", "beneficial", "advantageous"],
        "guru": ["expert", "specialist", "authority"],
        "wizard": ["expert", "skilled professional", "technical specialist"],
        "competitive": ["motivated", "results-focused", "driven"],
        "forceful": ["decisive", "determined", "confident"]
    }
}


class JSONBasedBiasAnalyzer:
    """JSON-based bias analyzer without external dependencies"""

    def __init__(self, bias_config: Dict):
        """Initialize analyzer with JSON configuration"""
        self.config = bias_config

        # Flatten word lists
        self.masculine_words = self._flatten_categories(bias_config.get('masculine_coded', {}))
        self.feminine_words = self._flatten_categories(bias_config.get('feminine_coded', {}))
        self.inclusive_words = self._flatten_categories(bias_config.get('inclusive_terms', {}))
        self.exclusive_words = self._flatten_categories(bias_config.get('exclusive_indicators', {}))

        # Get replacement rules
        self.neutral_alternatives = bias_config.get('neutral_alternatives', {})

        # Add replacement words to inclusive words for scoring
        replacement_words = set()
        for alternatives in self.neutral_alternatives.values():
            replacement_words.update(alternatives)
        self.inclusive_words.extend(list(replacement_words))

        # Remove duplicates and convert to lowercase for matching
        self.masculine_words = [word.lower() for word in set(self.masculine_words)]
        self.feminine_words = [word.lower() for word in set(self.feminine_words)]
        self.inclusive_words = [word.lower() for word in set(self.inclusive_words)]
        self.exclusive_words = [word.lower() for word in set(self.exclusive_words)]

    def _flatten_categories(self, categories: Dict) -> List[str]:
        """Flatten word categories into a single list"""
        words = []
        for category_words in categories.values():
            words.extend(category_words)
        return words

    def analyze_text(self, text: str) -> Dict:
        """Analyze text for bias patterns"""
        text_lower = text.lower()

        # Find words using more flexible matching
        found_masculine = self._find_words_in_text(text_lower, self.masculine_words)
        found_feminine = self._find_words_in_text(text_lower, self.feminine_words)
        found_inclusive = self._find_words_in_text(text_lower, self.inclusive_words)
        found_exclusive = self._find_words_in_text(text_lower, self.exclusive_words)

        # Calculate bias direction
        masculine_count = len(found_masculine)
        feminine_count = len(found_feminine)
        inclusive_count = len(found_inclusive)

        if masculine_count > inclusive_count and masculine_count > feminine_count:
            bias_direction = "Masculine"
        elif feminine_count > masculine_count and feminine_count > inclusive_count:
            bias_direction = "Feminine"
        elif inclusive_count > masculine_count:
            bias_direction = "Inclusive"
        else:
            bias_direction = "Neutral"

        # Calculate inclusivity score with replacement bonus
        total_words = len(text.split())
        bias_penalty = (masculine_count + len(found_exclusive)) * 8
        inclusive_bonus = inclusive_count * 12

        # Check for successful replacements (words that were replaced)
        replacement_bonus = self._calculate_replacement_bonus(text_lower)

        base_score = max(0, 75 - bias_penalty + inclusive_bonus + replacement_bonus)
        inclusivity_score = min(100, base_score)

        # Predict women application rate
        women_rate = max(15, min(85, 55 - masculine_count * 4 + inclusive_count * 6 + replacement_bonus * 0.5))

        return {
            'masculine_words': found_masculine,
            'feminine_words': found_feminine,
            'inclusive_words': found_inclusive,
            'exclusive_words': found_exclusive,
            'bias_direction': bias_direction,
            'inclusivity_score': inclusivity_score,
            'women_application_rate': women_rate,
            'replacement_bonus': replacement_bonus,
            'recommendations': self._generate_recommendations(
                found_masculine, found_inclusive, found_exclusive
            )
        }

    def _find_words_in_text(self, text: str, word_list: List[str]) -> List[str]:
        """Find words in text using flexible matching"""
        found = []
        for word in word_list:
            # Use word boundary matching for better accuracy
            if re.search(r'\b' + re.escape(word) + r'\b', text, re.IGNORECASE):
                found.append(word)
        return found

    def _calculate_replacement_bonus(self, text_lower: str) -> float:
        """Calculate bonus score for using replacement words"""
        bonus = 0
        for original, alternatives in self.neutral_alternatives.items():
            # Check if original word is NOT in text (good)
            if not re.search(r'\b' + re.escape(original) + r'\b', text_lower):
                # Check if any alternative IS in text (even better)
                for alt in alternatives:
                    if re.search(r'\b' + re.escape(alt.lower()) + r'\b', text_lower):
                        bonus += 5  # Bonus for each successful replacement
                        break
        return bonus

    def _generate_recommendations(self, masculine, inclusive, exclusive) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []

        if len(masculine) > 2:
            # Suggest specific replacements from our config
            suggested_replacements = []
            for word in masculine[:3]:  # Show first 3
                if word in self.neutral_alternatives:
                    alt = self.neutral_alternatives[word][0]  # First alternative
                    suggested_replacements.append(f"'{word}' → '{alt}'")

            if suggested_replacements:
                recommendations.append(
                    f"Consider replacing: {', '.join(suggested_replacements)}"
                )

        if len(inclusive) < 3:
            recommendations.append(
                "Add more inclusive language like 'collaborative', 'supportive', 'diverse'"
            )

        if len(exclusive) > 0:
            recommendations.append(
                f"Soften exclusive requirements: {', '.join(exclusive[:2])}"
            )

        if not recommendations:
            recommendations.append("This job description has good gender balance!")

        return recommendations

    def get_intelligent_rewrite(self, text: str) -> Dict:
        """Generate intelligent rewrite suggestions"""
        improved_text = text
        changes_made = []

        # Apply replacements from JSON config
        for original, alternatives in self.neutral_alternatives.items():
            pattern = r'\b' + re.escape(original) + r'\b'
            if re.search(pattern, improved_text, re.IGNORECASE):
                replacement = alternatives[0]  # Use first alternative
                improved_text = re.sub(pattern, replacement, improved_text, flags=re.IGNORECASE)
                changes_made.append(f"'{original}' → '{replacement}'")

        # Add inclusive language if needed
        original_analysis = self.analyze_text(text)
        if len(original_analysis['inclusive_words']) < 2:
            if 'collaborative' not in improved_text.lower():
                improved_text += " We value collaborative team members."
                changes_made.append("Added: 'We value collaborative team members'")

        return {
            'improved_text': improved_text,
            'changes_made': changes_made,
            'original_analysis': original_analysis,
            'improved_analysis': self.analyze_text(improved_text)
        }


def highlight_text(text: str, analysis: Dict) -> str:
    """Apply highlighting to text"""
    highlighted = text

    # Highlight masculine words
    for word in analysis['masculine_words']:
        pattern = r'\b' + re.escape(word) + r'\b'
        highlighted = re.sub(
            pattern,
            f'<span class="highlight-masculine">{word}</span>',
            highlighted,
            flags=re.IGNORECASE
        )

    # Highlight feminine words
    for word in analysis['feminine_words']:
        pattern = r'\b' + re.escape(word) + r'\b'
        highlighted = re.sub(
            pattern,
            f'<span class="highlight-inclusive">{word}</span>',
            highlighted,
            flags=re.IGNORECASE
        )

    # Highlight inclusive words
    for word in analysis['inclusive_words']:
        pattern = r'\b' + re.escape(word) + r'\b'
        highlighted = re.sub(
            pattern,
            f'<span class="highlight-inclusive">{word}</span>',
            highlighted,
            flags=re.IGNORECASE
        )

    # Highlight exclusive words
    for word in analysis['exclusive_words']:
        pattern = r'\b' + re.escape(word) + r'\b'
        highlighted = re.sub(
            pattern,
            f'<span class="highlight-exclusive">{word}</span>',
            highlighted,
            flags=re.IGNORECASE
        )

    return highlighted


def highlight_comparison_text(text: str, changes_made: List[str], is_original: bool = True) -> str:
    """Apply highlighting for before/after comparison"""
    highlighted = text

    # Parse changes to extract original and replacement words
    for change in changes_made:
        if "→" in change:
            # Extract original and replacement from format: 'original' → 'replacement'
            parts = change.split("→")
            if len(parts) == 2:
                original = parts[0].strip().strip("'\"")
                replacement = parts[1].strip().strip("'\"")

                if is_original:
                    # Highlight original words with strikethrough in original text
                    pattern = r'\b' + re.escape(original) + r'\b'
                    highlighted = re.sub(
                        pattern,
                        f'<span style="background-color: #fecaca; text-decoration: line-through; color: #991b1b; padding: 2px 4px; border-radius: 3px;">{original}</span>',
                        highlighted,
                        flags=re.IGNORECASE
                    )
                else:
                    # Highlight replacement words in improved text
                    pattern = r'\b' + re.escape(replacement) + r'\b'
                    highlighted = re.sub(
                        pattern,
                        f'<span class="highlight-replacement">{replacement}</span>',
                        highlighted,
                        flags=re.IGNORECASE
                    )

    # For added phrases (not replacements)
    if not is_original:
        for change in changes_made:
            if "Added:" in change:
                # Extract added phrase
                added_phrase = change.replace("Added:", "").strip().strip("'\"")
                if added_phrase in highlighted:
                    highlighted = highlighted.replace(
                        added_phrase,
                        f'<span style="background-color: #bfdbfe; color: #1e40af; padding: 2px 4px; border-radius: 3px; font-weight: bold;">{added_phrase}</span>'
                    )

    return highlighted


def main():
    """Main application"""

    # Header
    st.markdown("""
    <div class="header-box">
        <h1>🔍GENIA @ P4G</h1>
        <h3>Gender-Inclusive AI Assistant</h3>
        <p>Analyze job descriptions for gender bias with configurable rules</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize analyzer with JSON config
    if 'bias_config' not in st.session_state:
        st.session_state.bias_config = DEFAULT_BIAS_CONFIG

    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = JSONBasedBiasAnalyzer(st.session_state.bias_config)

    # Initialize analysis trigger
    if 'analysis_triggered' not in st.session_state:
        st.session_state.analysis_triggered = False

    # Sidebar
    with st.sidebar:
        st.header("📝 Analysis Mode")

        mode = st.selectbox("Choose Mode", [
            "Single Analysis",
            "Demo Examples",
            "Configure JSON"
        ])

        if mode == "Demo Examples":
            demo_option = st.selectbox("Select Demo", [
                "Biased Example",
                "Inclusive Example"
            ])
        elif mode == "Configure JSON":
            st.subheader("📋 JSON Configuration")

            # JSON config editor
            config_editor = st.text_area(
                "Edit Bias Configuration:",
                value=json.dumps(st.session_state.bias_config, indent=2),
                height=300,
                help="Edit the JSON configuration to customize word lists and replacement rules"
            )

            if st.button("🔄 Update Configuration"):
                try:
                    new_config = json.loads(config_editor)
                    st.session_state.bias_config = new_config
                    st.session_state.analyzer = JSONBasedBiasAnalyzer(new_config)
                    st.success("✅ Configuration updated successfully!")
                except json.JSONDecodeError as e:
                    st.error(f"❌ Invalid JSON: {e}")

            if st.button("🔄 Reset to Default"):
                st.session_state.bias_config = DEFAULT_BIAS_CONFIG
                st.session_state.analyzer = JSONBasedBiasAnalyzer(DEFAULT_BIAS_CONFIG)
                st.success("✅ Configuration reset to default!")

        # Clear button
        if st.button("🗑️ Clear All"):
            for key in list(st.session_state.keys()):
                if key not in ['bias_config', 'analyzer']:  # Keep configuration
                    del st.session_state[key]
            st.session_state.analysis_triggered = False

    # Main content
    if mode == "Configure JSON":
        st.subheader("🔧 JSON Configuration Editor")
        st.info("Use the sidebar to edit the JSON configuration for word lists and replacement rules.")

        # Display current config statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Masculine Words", len(st.session_state.analyzer.masculine_words))
        with col2:
            st.metric("Inclusive Words", len(st.session_state.analyzer.inclusive_words))
        with col3:
            st.metric("Exclusive Words", len(st.session_state.analyzer.exclusive_words))
        with col4:
            st.metric("Replacement Rules", len(st.session_state.analyzer.neutral_alternatives))

    elif mode == "Demo Examples":
        demo_texts = {
            "Biased Example": "We need an aggressive ninja developer with strong leadership skills who can dominate the competition and crush deadlines. This demanding role requires a rockstar who can work independently.",
            "Inclusive Example": "We welcome a collaborative developer to join our diverse team. We offer flexible work arrangements, mentorship opportunities, and value supportive team members who contribute to our inclusive environment."
        }

        demo_text = demo_texts[demo_option]
        st.text_area("Demo Job Description", demo_text, height=100, disabled=True, key="demo_text")

        if st.button("🔍 Analyze Demo", type="primary"):
            with st.spinner("Analyzing..."):
                analysis = st.session_state.analyzer.analyze_text(demo_text)
                st.session_state.current_analysis = analysis
                st.session_state.current_text = demo_text
                st.session_state.analysis_triggered = True
                st.success("✅ Analysis completed!")

    else:
        # Normal analysis mode
        st.subheader("📝 Enter Job Description")

        job_text = st.text_area(
            "Paste your job description here:",
            height=150,
            placeholder="Enter the job description to analyze...",
            key="job_input"
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔍 Analyze", type="primary", disabled=not job_text.strip()):
                if job_text.strip():
                    with st.spinner("Analyzing..."):
                        analysis = st.session_state.analyzer.analyze_text(job_text)
                        st.session_state.current_analysis = analysis
                        st.session_state.current_text = job_text
                        st.session_state.analysis_triggered = True
                        st.success("✅ Analysis completed!")

        with col2:
            if st.button("🗑️ Clear"):
                if 'current_analysis' in st.session_state:
                    del st.session_state.current_analysis
                if 'current_text' in st.session_state:
                    del st.session_state.current_text
                st.session_state.analysis_triggered = False

    # Display results
    if (st.session_state.get('analysis_triggered') and
            'current_analysis' in st.session_state and
            'current_text' in st.session_state):

        st.markdown("---")

        analysis = st.session_state.current_analysis
        text = st.session_state.current_text

        # Display metrics
        st.subheader("📊 Analysis Results")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-box">
                <h4>Inclusivity Score</h4>
                <h2 style="color: #059669;">{analysis['inclusivity_score']:.0f}/100</h2>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-box">
                <h4>Women Applicants</h4>
                <h2 style="color: #3b82f6;">{analysis['women_application_rate']:.0f}%</h2>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-box">
                <h4>Bias Direction</h4>
                <h2 style="color: #6b7280;">{analysis['bias_direction']}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            replacement_bonus = analysis.get('replacement_bonus', 0)
            st.markdown(f"""
            <div class="metric-box">
                <h4>Replacement Bonus</h4>
                <h2 style="color: #10b981;">+{replacement_bonus:.0f}</h2>
            </div>
            """, unsafe_allow_html=True)

        # Tabs for detailed analysis
        tab1, tab2, tab3 = st.tabs(["🎨 Highlighted Text", "📋 Word Analysis", "💡 Smart Rewriting"])

        with tab1:
            st.subheader("Job Description with Highlighting")

            # Legend
            st.markdown("""
            **Legend:**
            <span class="highlight-masculine">Masculine words</span> • 
            <span class="highlight-inclusive">Inclusive words</span> • 
            <span class="highlight-exclusive">Exclusive words</span> •
            <span class="highlight-replacement">Replacement words</span>
            """, unsafe_allow_html=True)

            # Highlighted text
            highlighted_text = highlight_text(text, analysis)
            st.markdown(f'<div class="text-display">{highlighted_text}</div>', unsafe_allow_html=True)

        with tab2:
            st.subheader("Detected Words")

            col1, col2 = st.columns(2)

            with col1:
                if analysis['masculine_words']:
                    st.markdown("**🔴 Masculine Words:**")
                    for word in analysis['masculine_words']:
                        st.write(f"• {word}")

                if analysis['inclusive_words']:
                    st.markdown("**🔵 Inclusive Words:**")
                    for word in analysis['inclusive_words']:
                        st.write(f"• {word}")

            with col2:
                if analysis['exclusive_words']:
                    st.markdown("**🟠 Exclusive Words:**")
                    for word in analysis['exclusive_words']:
                        st.write(f"• {word}")

                # Summary stats
                st.markdown("**📈 Statistics:**")
                st.write(f"• Total masculine words: {len(analysis['masculine_words'])}")
                st.write(f"• Total inclusive words: {len(analysis['inclusive_words'])}")
                st.write(f"• Total exclusive words: {len(analysis['exclusive_words'])}")
                st.write(f"• Replacement bonus: +{analysis.get('replacement_bonus', 0)}")

        with tab3:
            st.subheader("Smart Rewriting with JSON Rules")

            for i, recommendation in enumerate(analysis['recommendations'], 1):
                st.write(f"{i}. {recommendation}")

            # JSON-based intelligent rewriting
            if analysis['inclusivity_score'] < 75:
                st.warning("💡 Consider rewriting this job description to be more inclusive")

                # Get intelligent rewrite using JSON config
                rewrite_result = st.session_state.analyzer.get_intelligent_rewrite(text)
                improved_text = rewrite_result['improved_text']
                changes_made = rewrite_result['changes_made']
                improved_analysis = rewrite_result['improved_analysis']

                if improved_text != text and changes_made:
                    st.markdown("### Improvement")

                    # Show metrics comparison
                    st.markdown("#### Impact Analysis")
                    col_metric1, col_metric2, col_metric3 = st.columns(3)

                    with col_metric1:
                        score_change = improved_analysis['inclusivity_score'] - analysis['inclusivity_score']
                        st.metric(
                            "Inclusivity Score",
                            f"{improved_analysis['inclusivity_score']:.0f}/100",
                            f"{score_change:+.0f}" if score_change != 0 else "No change"
                        )

                    with col_metric2:
                        rate_change = improved_analysis['women_application_rate'] - analysis['women_application_rate']
                        st.metric(
                            "Women Applicants",
                            f"{improved_analysis['women_application_rate']:.0f}%",
                            f"{rate_change:+.0f}%" if rate_change != 0 else "No change"
                        )

                    with col_metric3:
                        bonus_change = improved_analysis.get('replacement_bonus', 0) - analysis.get('replacement_bonus',
                                                                                                    0)
                        st.metric(
                            "Replacement Bonus",
                            f"+{improved_analysis.get('replacement_bonus', 0):.0f}",
                            f"{bonus_change:+.0f}" if bonus_change != 0 else "No change"
                        )

                    # Text comparison
                    st.markdown("#### Before vs After")

                    col_orig, col_improved = st.columns(2)

                    with col_orig:
                        st.markdown("**📋 Original Version**")
                        # Highlight original text with strikethrough for replaced words
                        highlighted_original = highlight_comparison_text(text, changes_made, is_original=True)
                        st.markdown(f'<div class="text-display">{highlighted_original}</div>', unsafe_allow_html=True)

                    with col_improved:
                        st.markdown("**✨ Improved Version**")
                        # Highlight improved text with replacement words highlighted
                        highlighted_improved = highlight_comparison_text(improved_text, changes_made, is_original=False)
                        st.markdown(
                            f'<div class="text-display" style="border-left: 4px solid #10b981;">{highlighted_improved}</div>',
                            unsafe_allow_html=True)

                    # Show changes made
                    st.markdown("#### 🔍 Changes")
                    for i, change in enumerate(changes_made, 1):
                        st.write(f"{i}. {change}")

                    # Analyze improved version button
                    if st.button("🔍 Analyze Improved Version"):
                        st.session_state.current_analysis = improved_analysis
                        st.session_state.current_text = improved_text
                        st.session_state.analysis_triggered = True
                        st.success("✅ Switched to improved version analysis!")

    # Footer
    st.markdown("---")
    st.markdown("**Gender-Inclusive AI Assistant")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please refresh the page to restart the application.")

        # Debug info
        with st.expander("Debug Information"):
            import traceback

            st.code(traceback.format_exc())