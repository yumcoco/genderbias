"""
Minimal Gender Bias Analysis App
Simplified version to avoid recursion issues
"""

import streamlit as st
import re
from typing import Dict, List, Optional
from core.bias_detector import BiasDetector

# Set page config
st.set_page_config(
    page_title="Gender Bias Analysis",
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

# Built-in word lists to avoid external dependencies
MASCULINE_WORDS = [
    'aggressive', 'assertive', 'competitive', 'dominant', 'ambitious',
    'independent', 'leader', 'outgoing', 'confident', 'strong',
    'analytical', 'decisive', 'objective', 'challenging', 'demanding',
    'driven', 'determined', 'fearless', 'ninja', 'rockstar'
]

INCLUSIVE_WORDS = [
    'collaborative', 'cooperative', 'supportive', 'team', 'together',
    'diverse', 'inclusive', 'welcome', 'encourage', 'flexible',
    'work-life balance', 'mentorship', 'growth', 'learning'
]

EXCLUSIVE_WORDS = [
    'must have', 'required', 'mandatory', 'expert', 'guru',
    'dominate', 'crush', 'kill', 'destroy', 'annihilate'
]


class SimpleBiasAnalyzer:
    """Simple bias analyzer without external dependencies"""

    def __init__(self):
        self.masculine_words = MASCULINE_WORDS
        self.inclusive_words = INCLUSIVE_WORDS
        self.exclusive_words = EXCLUSIVE_WORDS

    def analyze_text(self, text: str) -> Dict:
        """Analyze text for bias patterns"""
        text_lower = text.lower()

        # Find words
        found_masculine = [word for word in self.masculine_words if word in text_lower]
        found_inclusive = [word for word in self.inclusive_words if word in text_lower]
        found_exclusive = [word for word in self.exclusive_words if word in text_lower]

        # Calculate bias direction
        masculine_count = len(found_masculine)
        inclusive_count = len(found_inclusive)

        if masculine_count > inclusive_count:
            bias_direction = "Masculine"
        elif inclusive_count > masculine_count:
            bias_direction = "Inclusive"
        else:
            bias_direction = "Neutral"

        # Calculate inclusivity score
        total_words = len(text.split())
        bias_penalty = (masculine_count + len(found_exclusive)) * 10
        inclusive_bonus = inclusive_count * 15
        base_score = max(0, 70 - bias_penalty + inclusive_bonus)
        inclusivity_score = min(100, base_score)

        # Predict women application rate
        women_rate = max(10, min(80, 50 - masculine_count * 5 + inclusive_count * 8))

        return {
            'masculine_words': found_masculine,
            'inclusive_words': found_inclusive,
            'exclusive_words': found_exclusive,
            'bias_direction': bias_direction,
            'inclusivity_score': inclusivity_score,
            'women_application_rate': women_rate,
            'recommendations': self._generate_recommendations(
                found_masculine, found_inclusive, found_exclusive
            )
        }

    def _generate_recommendations(self, masculine, inclusive, exclusive) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []

        if len(masculine) > 2:
            recommendations.append(
                f"Consider replacing masculine-coded words: {', '.join(masculine[:3])}"
            )

        if len(inclusive) < 2:
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


def main():
    """Main application"""

    # Header
    st.markdown("""
    <div class="header-box">
        <h1>🔍 Gender Bias Analysis</h1>
        <h3>Inclusive Hiring Assistant</h3>
        <p>Analyze job descriptions for gender bias</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize analyzer
    use_hf = st.sidebar.checkbox("Use HuggingFace Lexicon", value=True) # user can choose whether to start HuggingFace mode

    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = BiasDetector(use_hf=use_hf)



    # Initialize analysis trigger
    if 'analysis_triggered' not in st.session_state:
        st.session_state.analysis_triggered = False

    # Sidebar
    with st.sidebar:
        st.header("📝 Analysis Mode")

        mode = st.selectbox("Choose Mode", ["Single Analysis", "Demo Examples"])

        if mode == "Demo Examples":
            demo_option = st.selectbox("Select Demo", [
                "Biased Example",
                "Inclusive Example"
            ])

        # Clear button
        if st.button("Clear All"):
            for key in list(st.session_state.keys()):
                if key != 'analyzer':  # Keep the analyzer
                    del st.session_state[key]
            st.session_state.analysis_triggered = False

    # Main content
    if mode == "Demo Examples":
        demo_texts = {
            "Biased Example": "We need an aggressive ninja developer with strong leadership skills who can dominate the competition and crush deadlines.",
            "Inclusive Example": "We welcome a collaborative developer to join our diverse team. We offer flexible work arrangements and mentorship opportunities."
        }

        demo_text = demo_texts[demo_option]
        st.text_area("Demo Job Description", demo_text, height=100, disabled=True, key="demo_text")

        if st.button("🔍 Analyze Demo", type="primary"):
            with st.spinner("Analyzing..."):
                analysis = st.session_state.analyzer.analyze_bias_patterns(demo_text)
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
                        analysis = st.session_state.analyzer.analyze_bias_patterns(job_text)
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

    # Display results - 现在会立即显示
    if st.session_state.get(
            'analysis_triggered') and 'current_analysis' in st.session_state and 'current_text' in st.session_state:
        st.markdown("---")

        analysis = st.session_state.current_analysis
        text = st.session_state.current_text

        # Display metrics
        st.subheader("📊 Analysis Results")

        col1, col2, col3 = st.columns(3)

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

        # Tabs for detailed analysis
        tab1, tab2, tab3 = st.tabs(["🎨 Highlighted Text", "📋 Word Analysis", "💡 Recommendations"])

        with tab1:
            st.subheader("Job Description with Highlighting")

            # Legend
            st.markdown("""
            **Legend:**
            <span class="highlight-masculine">Masculine words</span> • 
            <span class="highlight-inclusive">Inclusive words</span> • 
            <span class="highlight-exclusive">Exclusive words</span>
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

        with tab3:
            st.subheader("Improvement Recommendations")

            for i, recommendation in enumerate(analysis['recommendations'], 1):
                st.write(f"{i}. {recommendation}")

            # Simple rewriter suggestion
            if analysis['inclusivity_score'] < 70:
                st.warning("💡 Consider rewriting this job description to be more inclusive")

                improved_text = text
                # Simple replacements
                replacements = {
                    'aggressive': 'proactive',
                    'ninja': 'skilled',
                    'rockstar': 'talented',
                    'dominate': 'excel in',
                    'crush': 'meet',
                    'must have': 'preferred',
                    'required': 'desired'
                }

                for old, new in replacements.items():
                    improved_text = re.sub(
                        r'\b' + re.escape(old) + r'\b',
                        new,
                        improved_text,
                        flags=re.IGNORECASE
                    )

                # Add inclusive language
                if 'collaborative' not in improved_text.lower():
                    improved_text += " We value collaborative team members."

                if improved_text != text:
                    st.markdown("### 📝 Before vs After Comparison")

                    # 分析改进版本以获取新指标
                    improved_analysis = st.session_state.analyzer.analyze_bias_patterns(improved_text)

                    # 显示指标对比
                    st.markdown("#### 📊 Impact Analysis")
                    col_metric1, col_metric2, col_metric3 = st.columns(3)

                    with col_metric1:
                        score_change = improved_analysis['inclusivity_score'] - analysis['inclusivity_score']
                        st.metric(
                            "Inclusivity Score",
                            f"{improved_analysis['inclusivity_score']:.0f}/100",
                            f"{score_change:+.0f}" if score_change != 0 else "No change",
                            delta_color="normal"
                        )

                    with col_metric2:
                        rate_change = improved_analysis['women_application_rate'] - analysis['women_application_rate']
                        st.metric(
                            "Women Applicants",
                            f"{improved_analysis['women_application_rate']:.0f}%",
                            f"{rate_change:+.0f}%" if rate_change != 0 else "No change",
                            delta_color="normal"
                        )

                    with col_metric3:
                        # 计算偏见方向改善
                        bias_improvement = "✅ Improved" if improved_analysis['bias_direction'] != analysis[
                            'bias_direction'] else "Same"
                        st.metric(
                            "Bias Direction",
                            improved_analysis['bias_direction'],
                            bias_improvement if bias_improvement != "Same" else None
                        )

                    # 文本对比显示
                    st.markdown("#### 📝 Text Comparison")

                    col_orig, col_improved = st.columns(2)

                    with col_orig:
                        st.markdown("**📋 Original Version**")
                        # 高亮原文中被替换的词汇
                        highlighted_original = text
                        replacements = {
                            'aggressive': 'proactive',
                            'ninja': 'skilled',
                            'rockstar': 'talented',
                            'dominate': 'excel in',
                            'crush': 'meet',
                            'must have': 'preferred',
                            'required': 'desired'
                        }

                        for old, new in replacements.items():
                            if old.lower() in text.lower():
                                pattern = r'\b' + re.escape(old) + r'\b'
                                highlighted_original = re.sub(
                                    pattern,
                                    f'<span style="background-color: #fecaca; text-decoration: line-through; color: #991b1b; padding: 2px 4px; border-radius: 3px;">{old}</span>',
                                    highlighted_original,
                                    flags=re.IGNORECASE
                                )

                        st.markdown(f'<div class="text-display">{highlighted_original}</div>', unsafe_allow_html=True)

                        # 原始指标
                        st.markdown("**Original Metrics:**")
                        st.write(f"• Inclusivity Score: {analysis['inclusivity_score']:.0f}/100")
                        st.write(f"• Women Applicants: {analysis['women_application_rate']:.0f}%")
                        st.write(f"• Bias Direction: {analysis['bias_direction']}")

                    with col_improved:
                        st.markdown("**✨ Improved Version**")
                        # 高亮改进文本中的新词汇
                        highlighted_improved = improved_text

                        for old, new in replacements.items():
                            if new.lower() in improved_text.lower():
                                pattern = r'\b' + re.escape(new) + r'\b'
                                highlighted_improved = re.sub(
                                    pattern,
                                    f'<span style="background-color: #dcfce7; color: #166534; padding: 2px 4px; border-radius: 3px; font-weight: bold;">{new}</span>',
                                    highlighted_improved,
                                    flags=re.IGNORECASE
                                )

                        # 高亮新增的包容性语言
                        if 'collaborative' in improved_text.lower() and 'collaborative' not in text.lower():
                            highlighted_improved = re.sub(
                                r'\bcollaborative\b',
                                '<span style="background-color: #bfdbfe; color: #1e40af; padding: 2px 4px; border-radius: 3px; font-weight: bold;">collaborative</span>',
                                highlighted_improved,
                                flags=re.IGNORECASE
                            )

                        st.markdown(
                            f'<div class="text-display" style="border-left: 4px solid #10b981;">{highlighted_improved}</div>',
                            unsafe_allow_html=True)

                        # 改进后指标
                        st.markdown("**Improved Metrics:**")
                        st.write(
                            f"• Inclusivity Score: {improved_analysis['inclusivity_score']:.0f}/100 ({score_change:+.0f})")
                        st.write(
                            f"• Women Applicants: {improved_analysis['women_application_rate']:.0f}% ({rate_change:+.0f}%)")
                        st.write(f"• Bias Direction: {improved_analysis['bias_direction']}")

                    # 显示具体改动
                    st.markdown("#### 🔍 Changes Made")
                    changes_made = []
                    for old, new in replacements.items():
                        if old.lower() in text.lower():
                            changes_made.append(f"'{old}' → '{new}'")

                    if 'collaborative' in improved_text.lower() and 'collaborative' not in text.lower():
                        changes_made.append("Added inclusive language: 'We value collaborative team members'")

                    if changes_made:
                        for i, change in enumerate(changes_made, 1):
                            st.write(f"{i}. {change}")
                    else:
                        st.info("No specific word replacements made.")

                    # 改进总结
                    if score_change > 0 or rate_change > 0:
                        st.success(
                            f"🎉 **Improvement Summary:** Inclusivity score increased by {score_change:.0f} points, potential women applicants increased by {rate_change:.0f}%")

                    # 分析改进版本按钮
                    if st.button("🔍 Analyze Improved Version"):
                        st.session_state.current_analysis = improved_analysis
                        st.session_state.current_text = improved_text
                        st.session_state.analysis_triggered = True
                        st.success("✅ Switched to improved version analysis!")
                        # 不需要 rerun，会自动更新显示

    # Footer
    st.markdown("---")
    st.markdown("**Gender Bias Analysis Tool** - Making hiring more inclusive")


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