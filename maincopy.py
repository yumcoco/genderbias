"""
GenderLens AI - Clean Main Application
Fixed version with proper function ordering
"""

import streamlit as st
import sys
import os
import pandas as pd
import re
import hashlib
from typing import Optional, Dict, Any

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import core modules
from core.bias_detector import get_bias_detector
from core.inclusivity_scorer import get_inclusivity_scorer
from core.prediction_model import get_women_predictor
from core.text_rewriter import get_text_rewriter
from utils.helpers import validate_text_input

# Page configuration
st.set_page_config(
    page_title="Decode Gender Bias",
    page_icon="🔍",
    layout="wide"
)

# Enhanced CSS with highlighting styles
st.markdown("""
<style>
    .header-box {
        background: #6366f1;
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
    }

    /* Bias word highlighting */
    .highlight-masculine {
        background-color: #fca5a5;
        color: #991b1b;
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: bold;
        border: 1px solid #f87171;
    }

    .highlight-feminine {
        background-color: #a7f3d0;
        color: #065f46;
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: bold;
        border: 1px solid #6ee7b7;
    }

    .highlight-inclusive {
        background-color: #bfdbfe;
        color: #1e40af;
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: bold;
        border: 1px solid #93c5fd;
    }

    .highlight-exclusive {
        background-color: #fed7aa;
        color: #9a3412;
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: bold;
        border: 1px solid #fdba74;
    }

    /* Text display boxes */
    .text-display-box {
        background: #f9fafb;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        line-height: 1.6;
        margin: 10px 0;
        font-size: 14px;
    }

    /* Legend styles */
    .legend-item {
        display: inline-block;
        margin-right: 15px;
        margin-bottom: 5px;
        font-size: 12px;
    }

    /* Improved text comparison */
    .comparison-container {
        background: white;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        overflow: hidden;
    }

    .comparison-header {
        background: #f3f4f6;
        padding: 10px 15px;
        font-weight: bold;
        border-bottom: 1px solid #e5e7eb;
    }

    .comparison-content {
        padding: 15px;
    }
</style>
""", unsafe_allow_html=True)


def safe_get_models():
    """Safely get models without caching decorators"""
    if 'models_initialized' not in st.session_state:
        try:
            with st.spinner("Loading models..."):
                st.session_state.bias_detector = get_bias_detector()
                st.session_state.inclusivity_scorer = get_inclusivity_scorer()
                st.session_state.women_predictor = get_women_predictor()
                st.session_state.text_rewriter = get_text_rewriter()
                st.session_state.models_initialized = True
            st.success("Models loaded successfully!")
        except Exception as e:
            st.error(f"Failed to load models: {str(e)}")
            return False
    return True


def highlight_bias_words(text: str, bias_analysis) -> str:
    """Apply color highlighting to bias words in text"""
    highlighted_text = text

    # Highlight masculine words (red)
    if hasattr(bias_analysis, 'masculine_words') and bias_analysis.masculine_words:
        for word in bias_analysis.masculine_words:
            pattern = r'\b' + re.escape(word) + r'\b'
            highlighted_text = re.sub(
                pattern,
                f'<span class="highlight-masculine">{word}</span>',
                highlighted_text,
                flags=re.IGNORECASE
            )

    # Highlight feminine words (green)
    if hasattr(bias_analysis, 'feminine_words') and bias_analysis.feminine_words:
        for word in bias_analysis.feminine_words:
            pattern = r'\b' + re.escape(word) + r'\b'
            highlighted_text = re.sub(
                pattern,
                f'<span class="highlight-feminine">{word}</span>',
                highlighted_text,
                flags=re.IGNORECASE
            )

    # Highlight inclusive words (blue)
    if hasattr(bias_analysis, 'inclusive_words') and bias_analysis.inclusive_words:
        for word in bias_analysis.inclusive_words:
            pattern = r'\b' + re.escape(word) + r'\b'
            highlighted_text = re.sub(
                pattern,
                f'<span class="highlight-inclusive">{word}</span>',
                highlighted_text,
                flags=re.IGNORECASE
            )

    # Highlight exclusive words (orange)
    if hasattr(bias_analysis, 'exclusive_words') and bias_analysis.exclusive_words:
        for word in bias_analysis.exclusive_words:
            pattern = r'\b' + re.escape(word) + r'\b'
            highlighted_text = re.sub(
                pattern,
                f'<span class="highlight-exclusive">{word}</span>',
                highlighted_text,
                flags=re.IGNORECASE
            )

    return highlighted_text


def highlight_rewrite_changes(original_text: str, rewritten_text: str, changes=None) -> tuple:
    """Highlight changes between original and rewritten text"""

    if changes:
        highlighted_original = original_text
        highlighted_rewritten = rewritten_text

        for change in changes:
            if hasattr(change, 'original') and hasattr(change, 'replacement'):
                original_word = change.original
                replacement_word = change.replacement

                if original_word and replacement_word and original_word != replacement_word:
                    # Highlight removed words in original (red background)
                    pattern = r'\b' + re.escape(original_word) + r'\b'
                    highlighted_original = re.sub(
                        pattern,
                        f'<span style="background-color: #fecaca; text-decoration: line-through; color: #991b1b; padding: 2px 4px; border-radius: 3px;">{original_word}</span>',
                        highlighted_original,
                        flags=re.IGNORECASE
                    )

                    # Highlight new words in rewritten (green background)
                    pattern = r'\b' + re.escape(replacement_word) + r'\b'
                    highlighted_rewritten = re.sub(
                        pattern,
                        f'<span style="background-color: #dcfce7; color: #166534; padding: 2px 4px; border-radius: 3px; font-weight: bold;">{replacement_word}</span>',
                        highlighted_rewritten,
                        flags=re.IGNORECASE
                    )

        return highlighted_original, highlighted_rewritten
    else:
        return original_text, rewritten_text


def display_legend():
    """Display color legend for bias word highlighting"""
    st.markdown("""
    <div style="margin: 10px 0; padding: 10px; background: #f9fafb; border-radius: 8px;">
        <strong>Legend:</strong><br>
        <span class="legend-item"><span class="highlight-masculine">Masculine words</span> - May discourage women applicants</span>
        <span class="legend-item"><span class="highlight-feminine">Feminine words</span> - May appeal more to women</span><br>
        <span class="legend-item"><span class="highlight-inclusive">Inclusive words</span> - Welcoming to all candidates</span>
        <span class="legend-item"><span class="highlight-exclusive">Exclusive words</span> - May create barriers</span>
    </div>
    """, unsafe_allow_html=True)


def simple_analyze(text: str) -> Optional[Dict[str, Any]]:
    """Simple analysis function without complex caching"""
    if not safe_get_models():
        return None

    try:
        with st.spinner("Analyzing..."):
            # Get bias analysis
            bias_analysis = st.session_state.bias_detector.analyze_bias_patterns(text)

            # Get inclusivity score
            inclusivity_score = st.session_state.inclusivity_scorer.score_job_description(text)

            # Get prediction
            prediction = st.session_state.women_predictor.predict_women_proportion(text)

            return {
                'text': text,
                'bias_analysis': bias_analysis,
                'inclusivity_score': inclusivity_score,
                'prediction': prediction
            }

    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        return None


def display_simple_header():
    """Simple header without complex markdown"""
    st.markdown("""
    <div class="header-box">
        <h1>Decode Gender Bias</h1>
        <h3>Inclusive Hiring Assistant</h3>
        <p>Analyze job descriptions for gender bias</p>
    </div>
    """, unsafe_allow_html=True)


def display_simple_metrics(analysis):
    """Display metrics in simple format"""
    col1, col2, col3 = st.columns(3)

    with col1:
        score = analysis['inclusivity_score'].overall_score
        st.markdown(f"""
        <div class="metric-box">
            <h4>Inclusivity Score</h4>
            <h2 style="color: #059669;">{score:.1f}/100</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        percentage = analysis['prediction']['percentage']
        st.markdown(f"""
        <div class="metric-box">
            <h4>Women Applicants</h4>
            <h2 style="color: #3b82f6;">{percentage:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        bias = analysis['bias_analysis'].overall_bias
        st.markdown(f"""
        <div class="metric-box">
            <h4>Bias Direction</h4>
            <h2 style="color: #6b7280;">{bias.title()}</h2>
        </div>
        """, unsafe_allow_html=True)


def display_simple_analysis(analysis):
    """Display analysis with highlighted text"""
    tab1, tab2, tab3 = st.tabs(["📊 Highlighted Analysis", "📝 Word Lists", "🚀 Recommendations"])

    with tab1:
        st.subheader("Job Description with Bias Highlighting")

        # Display legend first
        display_legend()

        # Get highlighted text
        bias = analysis['bias_analysis']
        highlighted_text = highlight_bias_words(analysis['text'], bias)

        # Display highlighted text
        st.markdown(f"""
        <div class="text-display-box">
            {highlighted_text}
        </div>
        """, unsafe_allow_html=True)

        # Word count summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            masculine_count = len(bias.masculine_words) if hasattr(bias, 'masculine_words') else 0
            st.metric("Masculine Words", masculine_count)
        with col2:
            feminine_count = len(bias.feminine_words) if hasattr(bias, 'feminine_words') else 0
            st.metric("Feminine Words", feminine_count)
        with col3:
            inclusive_count = len(bias.inclusive_words) if hasattr(bias, 'inclusive_words') else 0
            st.metric("Inclusive Words", inclusive_count)
        with col4:
            exclusive_count = len(bias.exclusive_words) if hasattr(bias, 'exclusive_words') else 0
            st.metric("Exclusive Words", exclusive_count)

    with tab2:
        st.subheader("Detected Word Categories")
        bias = analysis['bias_analysis']

        col1, col2 = st.columns(2)

        with col1:
            if hasattr(bias, 'masculine_words') and bias.masculine_words:
                st.markdown("**🔴 Masculine Words:**")
                for word in bias.masculine_words:
                    st.markdown(f"• <span class='highlight-masculine'>{word}</span>", unsafe_allow_html=True)

            if hasattr(bias, 'inclusive_words') and bias.inclusive_words:
                st.markdown("**🔵 Inclusive Words:**")
                for word in bias.inclusive_words:
                    st.markdown(f"• <span class='highlight-inclusive'>{word}</span>", unsafe_allow_html=True)

        with col2:
            if hasattr(bias, 'feminine_words') and bias.feminine_words:
                st.markdown("**🟢 Feminine Words:**")
                for word in bias.feminine_words:
                    st.markdown(f"• <span class='highlight-feminine'>{word}</span>", unsafe_allow_html=True)

            if hasattr(bias, 'exclusive_words') and bias.exclusive_words:
                st.markdown("**🟠 Exclusive Words:**")
                for word in bias.exclusive_words:
                    st.markdown(f"• <span class='highlight-exclusive'>{word}</span>", unsafe_allow_html=True)

        if not any([
            hasattr(bias, 'masculine_words') and bias.masculine_words,
            hasattr(bias, 'feminine_words') and bias.feminine_words,
            hasattr(bias, 'inclusive_words') and bias.inclusive_words,
            hasattr(bias, 'exclusive_words') and bias.exclusive_words
        ]):
            st.info("No significant bias words detected in this text.")

    with tab3:
        st.subheader("Improvement Recommendations")
        recommendations = analysis['inclusivity_score'].recommendations

        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
        else:
            st.success("✅ This job description is well-balanced!")


def main():
    """Main application function - simplified"""

    # Display header
    display_simple_header()

    # Sidebar
    with st.sidebar:
        st.header("Analysis Mode")

        mode = st.selectbox("Choose Mode", ["Single Analysis", "Demo Examples"])

        if mode == "Demo Examples":
            demo_option = st.selectbox("Select Demo", [
                "Biased Example",
                "Inclusive Example"
            ])

        # Clear button
        if st.button("Clear All"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # Main content
    if mode == "Demo Examples":
        demo_texts = {
            "Biased Example": "We need an aggressive ninja developer with strong leadership skills who can dominate the competition.",
            "Inclusive Example": "We welcome a collaborative developer to join our diverse team with flexible work arrangements."
        }

        demo_text = demo_texts[demo_option]
        st.text_area("Demo Job Description", demo_text, height=100, disabled=True, key="demo_text_area")

        if st.button("Analyze Demo", type="primary"):
            result = simple_analyze(demo_text)
            if result:
                st.session_state.current_analysis = result
                st.rerun()

    else:
        # Normal analysis mode
        st.subheader("Enter Job Description")

        job_text = st.text_area(
            "Paste your job description here:",
            height=150,
            placeholder="Enter the job description to analyze...",
            key="main_job_text_area"
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Analyze", type="primary", disabled=not job_text.strip()):
                if job_text.strip():
                    is_valid, message = validate_text_input(job_text)
                    if not is_valid:
                        st.error(message)
                    else:
                        result = simple_analyze(job_text)
                        if result:
                            st.session_state.current_analysis = result
                            st.rerun()

        with col2:
            if st.button("Clear"):
                if 'current_analysis' in st.session_state:
                    del st.session_state.current_analysis
                st.rerun()

    # Display results
    if 'current_analysis' in st.session_state:
        st.markdown("---")
        analysis = st.session_state.current_analysis

        # Display metrics
        display_simple_metrics(analysis)

        st.markdown("---")

        # Display analysis
        display_simple_analysis(analysis)

        # Rewriter section with enhanced comparison
        st.markdown("---")
        st.subheader("✨ AI Text Rewriter")

        score = analysis['inclusivity_score'].overall_score
        if score < 70:
            st.warning(f"⚠️ Inclusivity score is {score:.1f}/100. AI rewriting recommended.")
        else:
            st.info(f"✅ Good score ({score:.1f}/100), but AI can still optimize further.")

        if st.button("🤖 Generate Improved Version", type="primary"):
            try:
                with st.spinner("Rewriting text..."):
                    rewriter = st.session_state.text_rewriter
                    result = rewriter.intelligent_rewrite(analysis['text'])

                    if result:
                        st.success("✅ Rewriting completed!")

                        # Enhanced comparison with highlighting
                        st.markdown("### 📝 Before vs After Comparison")

                        # Get highlighted versions
                        original_highlighted, rewritten_highlighted = highlight_rewrite_changes(
                            result.original_text,
                            result.rewritten_text,
                            getattr(result, 'changes', None)
                        )

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("""
                            <div class="comparison-container">
                                <div class="comparison-header">📋 Original Text</div>
                                <div class="comparison-content">
                            """ + original_highlighted + """
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        with col2:
                            st.markdown("""
                            <div class="comparison-container">
                                <div class="comparison-header">✨ Improved Text</div>
                                <div class="comparison-content">
                            """ + rewritten_highlighted + """
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        # Show changes summary if available
                        if hasattr(result, 'changes') and result.changes:
                            st.markdown("### 🔍 Changes Made")

                            changes_to_show = [c for c in result.changes
                                               if hasattr(c, 'original') and hasattr(c, 'replacement')
                                               and c.original != c.replacement][:5]

                            if changes_to_show:
                                for i, change in enumerate(changes_to_show, 1):
                                    st.markdown(f"**{i}.** `{change.original}` → `{change.replacement}`")
                                    if hasattr(change, 'reason'):
                                        st.caption(f"Reason: {change.reason}")

                                if len(result.changes) > 5:
                                    st.caption(f"... and {len(result.changes) - 5} more changes")

                        # Action buttons
                        st.markdown("### 🎯 Next Steps")
                        col_btn1, col_btn2, col_btn3 = st.columns(3)

                        with col_btn1:
                            if st.button("🔍 Analyze Improved Version"):
                                improved_analysis = simple_analyze(result.rewritten_text)
                                if improved_analysis:
                                    st.session_state.current_analysis = improved_analysis
                                    st.success("✅ Re-analysis completed!")
                                    st.rerun()

                        with col_btn2:
                            # Download comparison
                            comparison_data = pd.DataFrame({
                                'Metric': ['Original Text', 'Improved Text', 'Changes Made'],
                                'Value': [result.original_text, result.rewritten_text,
                                          str(len(getattr(result, 'changes', [])))]
                            })
                            csv = comparison_data.to_csv(index=False)
                            st.download_button(
                                "📥 Download Comparison",
                                csv,
                                file_name="text_improvement_comparison.csv",
                                mime="text/csv"
                            )

                        with col_btn3:
                            if st.button("🔄 Try Alternative"):
                                try:
                                    with st.spinner("Generating alternative..."):
                                        alt_result = rewriter.intelligent_rewrite(analysis['text'])
                                        if alt_result:
                                            st.rerun()
                                except Exception as e:
                                    st.error(f"Failed to generate alternative: {str(e)}")

            except Exception as e:
                st.error(f"Rewriting failed: {str(e)}")

    # Simple footer
    st.markdown("---")
    st.markdown("**Decode Gender Bias** - Making hiring more inclusive")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Try refreshing the page or clearing the cache.")