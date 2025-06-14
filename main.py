"""
GenderLens AI - 主应用入口
智能招聘包容性分析与优化平台
"""

import streamlit as st
import sys
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入核心模块
from core.bias_detector import get_bias_detector
from core.inclusivity_scorer import get_inclusivity_scorer
from core.prediction_model import get_women_predictor
from core.text_rewriter import get_text_rewriter
from data.data_loader import get_data_loader
from ui.dashboard import create_main_dashboard
from ui.components import create_sidebar, display_results
from utils.helpers import validate_text_input

# 页面配置
st.set_page_config(
    page_title="GenderLens AI - Inclusive Hiring Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "GenderLens AI helps create more inclusive job descriptions by detecting gender bias and providing actionable recommendations."
    }
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }

    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }

    .suggestion-box {
        background: #f8f9ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }

    .warning-box {
        background: #fff8e1;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }

    .rewrite-box {
        background: #f0f8ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }

    .change-box {
        background: #f9f9f9;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #9C27B0;
        margin: 0.5rem 0;
    }

    .score-excellent { color: #4CAF50; font-weight: bold; }
    .score-good { color: #8BC34A; font-weight: bold; }
    .score-fair { color: #FFC107; font-weight: bold; }
    .score-poor { color: #FF9800; font-weight: bold; }
    .score-very-poor { color: #F44336; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """初始化会话状态"""
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    if 'current_analysis' not in st.session_state:
        st.session_state.current_analysis = None
    if 'demo_mode' not in st.session_state:
        st.session_state.demo_mode = False
    if 'rewrite_result' not in st.session_state:
        st.session_state.rewrite_result = None
    if 'show_rewrite' not in st.session_state:
        st.session_state.show_rewrite = False


def display_header():
    """显示应用头部"""
    st.markdown("""
    <div class="main-header">
        <h1>⚖️ GenderLens AI</h1>
        <h3>Intelligent Gender Bias Detection & Inclusive Hiring Assistant</h3>
        <p>Analyze job descriptions for gender bias and get actionable recommendations to create more inclusive postings</p>
    </div>
    """, unsafe_allow_html=True)


def display_metrics_overview():
    """显示关键指标概览"""
    if st.session_state.current_analysis:
        analysis = st.session_state.current_analysis

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            score = analysis['inclusivity_score'].overall_score
            score_class = get_score_css_class(score)
            st.markdown(f"""
            <div class="metric-card">
                <h4>Inclusivity Score</h4>
                <h2 class="{score_class}">{score:.1f}/100</h2>
                <p>{analysis['inclusivity_score'].grade}</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            prediction = analysis['prediction']['percentage']
            st.markdown(f"""
            <div class="metric-card">
                <h4>Predicted Women Applicants</h4>
                <h2 style="color: #667eea;">{prediction:.1f}%</h2>
                <p>Expected application rate</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            bias_direction = analysis['bias_analysis'].overall_bias.title()
            bias_color = "#FF9800" if bias_direction == "Masculine" else "#4CAF50" if bias_direction == "Feminine" else "#757575"
            st.markdown(f"""
            <div class="metric-card">
                <h4>Bias Direction</h4>
                <h2 style="color: {bias_color};">{bias_direction}</h2>
                <p>Overall language tendency</p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            total_suggestions = len(analysis['inclusivity_score'].recommendations)
            st.markdown(f"""
            <div class="metric-card">
                <h4>Recommendations</h4>
                <h2 style="color: #9C27B0;">{total_suggestions}</h2>
                <p>Improvement suggestions</p>
            </div>
            """, unsafe_allow_html=True)


def get_score_css_class(score):
    """根据分数获取CSS类名"""
    if score >= 80:
        return "score-excellent"
    elif score >= 65:
        return "score-good"
    elif score >= 50:
        return "score-fair"
    elif score >= 35:
        return "score-poor"
    else:
        return "score-very-poor"


def analyze_job_description(text):
    """分析职位描述的完整流程"""
    try:
        # 初始化分析器
        bias_detector = get_bias_detector()
        inclusivity_scorer = get_inclusivity_scorer()
        women_predictor = get_women_predictor()

        with st.spinner("Analyzing job description..."):
            # 执行分析
            bias_analysis = bias_detector.analyze_bias_patterns(text)
            inclusivity_score = inclusivity_scorer.score_job_description(text)
            prediction = women_predictor.predict_women_proportion(text)
            improvement_suggestions = women_predictor.generate_improvement_suggestions(text)

            # 构建分析结果
            analysis_result = {
                'text': text,
                'bias_analysis': bias_analysis,
                'inclusivity_score': inclusivity_score,
                'prediction': prediction,
                'improvement_suggestions': improvement_suggestions,
                'timestamp': pd.Timestamp.now()
            }

            return analysis_result

    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        return None


def display_detailed_analysis():
    """显示详细分析结果"""
    if not st.session_state.current_analysis:
        return

    analysis = st.session_state.current_analysis

    # 创建标签页 - 添加AI改写标签页
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Overview", "🔍 Bias Detection", "📈 Scoring Details", "🚀 Improvements", "✨ AI Rewriter"])

    with tab1:
        display_overview_tab(analysis)

    with tab2:
        display_bias_detection_tab(analysis)

    with tab3:
        display_scoring_details_tab(analysis)

    with tab4:
        display_improvements_tab(analysis)

    with tab5:
        display_rewriter_tab(analysis)


def display_overview_tab(analysis):
    """显示概览标签页"""
    st.subheader("📊 Analysis Overview")

    col1, col2 = st.columns([2, 1])

    with col1:
        # 主要发现
        st.markdown("#### Key Findings")

        bias = analysis['bias_analysis']
        score = analysis['inclusivity_score']

        if bias.overall_bias == 'masculine':
            st.markdown("""
            <div class="warning-box">
                <strong>⚠️ Masculine Language Detected</strong><br>
                This job description tends toward masculine-coded language, which may discourage women applicants.
            </div>
            """, unsafe_allow_html=True)
        elif bias.overall_bias == 'neutral':
            st.markdown("""
            <div class="suggestion-box">
                <strong>✅ Balanced Language</strong><br>
                The language in this job description is relatively gender-neutral.
            </div>
            """, unsafe_allow_html=True)

        # 显示优势
        if score.strengths:
            st.markdown("#### Strengths")
            for strength in score.strengths:
                st.markdown(f"✅ {strength}")

        # 显示劣势
        if score.weaknesses:
            st.markdown("#### Areas for Improvement")
            for weakness in score.weaknesses:
                st.markdown(f"⚠️ {weakness}")

    with col2:
        # 词汇统计图表
        st.markdown("#### Word Analysis")

        word_data = {
            'Category': ['Masculine', 'Feminine', 'Inclusive', 'Exclusive'],
            'Count': [
                len(bias.masculine_words),
                len(bias.feminine_words),
                len(bias.inclusive_words),
                len(bias.exclusive_words)
            ]
        }

        word_df = pd.DataFrame(word_data)

        # 创建更美观的条形图
        fig = px.bar(
            word_df,
            x='Category',
            y='Count',
            color='Category',
            color_discrete_map={
                'Masculine': '#FF5722',
                'Feminine': '#9C27B0',
                'Inclusive': '#4CAF50',
                'Exclusive': '#FF9800'
            },
            title="Word Distribution by Category"
        )
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def display_bias_detection_tab(analysis):
    """显示偏向检测标签页"""
    st.subheader("🔍 Gender Bias Detection")

    bias = analysis['bias_analysis']

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Masculine-Coded Words")
        if bias.masculine_words:
            for word in bias.masculine_words:
                st.markdown(f"🔸 **{word}**")
        else:
            st.markdown("✅ No masculine-coded words detected")

        st.markdown("#### Inclusive Words")
        if bias.inclusive_words:
            for word in bias.inclusive_words:
                st.markdown(f"🌟 **{word}**")
        else:
            st.markdown("⚠️ No inclusive words found")

    with col2:
        st.markdown("#### Feminine-Coded Words")
        if bias.feminine_words:
            for word in bias.feminine_words:
                st.markdown(f"💫 **{word}**")
        else:
            st.markdown("ℹ️ No feminine-coded words detected")

        st.markdown("#### Exclusive Language")
        if bias.exclusive_words:
            for word in bias.exclusive_words:
                st.markdown(f"⛔ **{word}**")
        else:
            st.markdown("✅ No exclusive language detected")


def display_scoring_details_tab(analysis):
    """显示评分详情标签页"""
    st.subheader("📈 Detailed Scoring")

    score = analysis['inclusivity_score']

    # 组件评分图表
    component_names = {
        'language_balance': 'Language Balance',
        'inclusivity': 'Inclusivity',
        'openness': 'Openness',
        'text_quality': 'Text Quality',
        'sentiment': 'Sentiment'
    }

    # 创建组件评分可视化
    components = []
    scores = []

    for component, score_value in score.component_scores.items():
        component_name = component_names.get(component, component)
        components.append(component_name)
        scores.append(score_value)

    fig = go.Figure(data=[
        go.Bar(
            x=components,
            y=scores,
            marker_color=['#4CAF50' if s >= 70 else '#FFC107' if s >= 50 else '#FF5722' for s in scores],
            text=[f'{s:.1f}' for s in scores],
            textposition='auto',
        )
    ])

    fig.update_layout(
        title="Component Scores Breakdown",
        xaxis_title="Components",
        yaxis_title="Score (0-100)",
        yaxis=dict(range=[0, 100]),
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # 详细进度条
    st.markdown("#### Component Details")
    for component, score_value in score.component_scores.items():
        component_name = component_names.get(component, component)

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{component_name}**")
            st.progress(score_value / 100)
        with col2:
            score_class = get_score_css_class(score_value)
            st.markdown(f'<span class="{score_class}">{score_value:.1f}/100</span>', unsafe_allow_html=True)


def display_improvements_tab(analysis):
    """显示改进建议标签页"""
    st.subheader("🚀 Improvement Recommendations")

    score = analysis['inclusivity_score']
    improvements = analysis['improvement_suggestions']

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### Specific Recommendations")
        for i, recommendation in enumerate(score.recommendations, 1):
            st.markdown(f"""
            <div class="suggestion-box">
                <strong>{i}.</strong> {recommendation}
            </div>
            """, unsafe_allow_html=True)

        if 'suggestions' in improvements and improvements['suggestions']:
            st.markdown("#### Impact-Based Suggestions")
            for suggestion in improvements['suggestions']:
                priority_color = "#FF5722" if suggestion['priority'] == 'high' else "#FF9800" if suggestion[
                                                                                                     'priority'] == 'medium' else "#4CAF50"
                st.markdown(f"""
                <div style="padding: 1rem; border-left: 4px solid {priority_color}; background: #f9f9f9; margin: 0.5rem 0; border-radius: 4px;">
                    <strong>{suggestion['suggestion']}</strong><br>
                    <small>Expected Impact: {suggestion['expected_impact']} | Priority: {suggestion['priority'].upper()}</small>
                </div>
                """, unsafe_allow_html=True)

    with col2:
        # 显示预期改进效果
        if 'estimated_improved_rate' in improvements:
            improved = improvements['estimated_improved_rate']
            current = improvements['current_rate']

            st.markdown("#### Estimated Impact")
            st.metric(
                "Current Women Application Rate",
                f"{current['percentage']:.1f}%"
            )
            st.metric(
                "Estimated Improved Rate",
                f"{improved['percentage']:.1f}%",
                f"+{improved['improvement']:.1f}%"
            )


def display_rewriter_tab(analysis):
    """显示AI改写标签页"""
    st.subheader("✨ AI-Powered Text Rewriter")

    # 改写功能介绍
    st.markdown("""
    Our intelligent rewriter automatically improves job descriptions based on bias analysis. 
    It uses evidence-based word replacements and adds inclusive language to boost your inclusivity score.
    """)

    score = analysis['inclusivity_score'].overall_score
    needs_rewrite = score < 70

    col1, col2 = st.columns([2, 1])

    with col1:
        # 改写建议
        if needs_rewrite:
            st.markdown(f"""
            <div class="warning-box">
                <strong>⚠️ AI Rewriting Recommended</strong><br>
                Your current score ({score:.1f}/100) suggests this job description could benefit from AI optimization.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="suggestion-box">
                <strong>✅ Good Score Detected</strong><br>
                Your score ({score:.1f}/100) is good, but AI can still help optimize further.
            </div>
            """, unsafe_allow_html=True)

        # 改写按钮
        if st.button("🤖 Generate AI-Improved Version", type="primary", key="ai_rewrite_main"):
            with st.spinner("AI is analyzing and rewriting your job description..."):
                try:
                    rewriter = get_text_rewriter()
                    rewrite_result = rewriter.intelligent_rewrite(analysis['text'])
                    st.session_state.rewrite_result = rewrite_result
                    st.session_state.show_rewrite = True
                    st.success("✅ AI rewriting completed! See results below.")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Rewriting failed: {str(e)}")

    with col2:
        st.markdown("#### ⚙️ Quick Settings")

        # 快速设置
        intensity = st.selectbox(
            "Rewrite Intensity:",
            ["Conservative", "Moderate", "Aggressive"],
            index=1,
            help="How extensively should AI modify the text?"
        )

        preserve_terms = st.checkbox(
            "Preserve Technical Terms",
            value=True,
            help="Keep industry-specific vocabulary unchanged"
        )

        # 快速操作按钮
        if st.button("📋 Copy Original Text"):
            st.info("Original text copied!")

        if st.session_state.rewrite_result:
            if st.button("📋 Copy Rewritten Text"):
                st.info("Rewritten text copied!")

    # 显示改写结果
    if st.session_state.show_rewrite and st.session_state.rewrite_result:
        display_rewrite_results()


def display_rewrite_results():
    """显示改写结果"""
    if not st.session_state.rewrite_result:
        return

    rewrite_result = st.session_state.rewrite_result

    st.markdown("---")
    st.markdown("### 📝 AI Rewriting Results")

    # 结果标签页
    result_tab1, result_tab2, result_tab3 = st.tabs(["📊 Before vs After", "🔍 Changes Made", "📋 Final Text"])

    with result_tab1:
        display_before_after_comparison(rewrite_result)

    with result_tab2:
        display_changes_made(rewrite_result)

    with result_tab3:
        display_final_text(rewrite_result)


def display_before_after_comparison(rewrite_result):
    """显示改写前后对比"""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📋 Original Text")
        st.text_area(
            "Original Job Description:",
            rewrite_result.original_text,
            height=200,
            disabled=True,
            key="original_text_display"
        )

        # 显示原文分析结果
        if st.session_state.current_analysis:
            analysis = st.session_state.current_analysis
            st.markdown("**Original Metrics:**")
            col1a, col1b = st.columns(2)
            with col1a:
                st.metric("Inclusivity Score", f"{analysis['inclusivity_score'].overall_score:.1f}")
            with col1b:
                st.metric("Women Rate", f"{analysis['prediction']['percentage']:.1f}%")

    with col2:
        st.markdown("#### ✨ AI-Rewritten Text")
        st.text_area(
            "Improved Job Description:",
            rewrite_result.rewritten_text,
            height=200,
            disabled=True,
            key="rewritten_text_display"
        )

        # 显示预测改进
        improvement = rewrite_result.improvement_prediction
        if 'new_score' in improvement and 'score_improvement' in improvement:
            st.markdown("**Predicted Improvements:**")
            col2a, col2b = st.columns(2)
            with col2a:
                st.metric(
                    "New Score",
                    f"{improvement['new_score']:.1f}",
                    f"+{improvement['score_improvement']:.1f}"
                )
            with col2b:
                # 估算新的女性申请率
                current_rate = st.session_state.current_analysis['prediction']['percentage']
                rate_improvement = improvement.get('predicted_women_rate_increase', 0)
                new_rate = current_rate + rate_improvement
                st.metric(
                    "Women Rate",
                    f"{new_rate:.1f}%",
                    f"+{rate_improvement:.1f}%"
                )


def display_changes_made(rewrite_result):
    """显示具体修改内容"""
    st.markdown("#### 🔍 Detailed Changes Made")

    if not rewrite_result.changes:
        st.info("✅ No changes were needed - your job description was already well-optimized!")
        return

    # 分类显示修改
    word_changes = [c for c in rewrite_result.changes if c.original and c.replacement != c.original]
    additions = [c for c in rewrite_result.changes if not c.original]

    if word_changes:
        st.markdown("**🔄 Word Replacements (Evidence-Based):**")
        for i, change in enumerate(word_changes, 1):
            st.markdown(f"""
            <div class="change-box">
                <strong>{i}. "{change.original}" → "{change.replacement}"</strong><br>
                <small><strong>Why:</strong> {change.reason}</small><br>
                <small><strong>Research:</strong> {change.evidence}</small>
            </div>
            """, unsafe_allow_html=True)

    if additions:
        st.markdown("**➕ Content Additions:**")
        for i, addition in enumerate(additions, 1):
            st.markdown(f"""
            <div class="suggestion-box">
                <strong>{i}. Added:</strong> "{addition.replacement}"<br>
                <small><strong>Purpose:</strong> {addition.reason}</small>
            </div>
            """, unsafe_allow_html=True)

    # 改进摘要
    st.markdown("**📈 Improvement Summary:**")
    improvement = rewrite_result.improvement_prediction

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Changes", len(rewrite_result.changes))
    with col2:
        if 'masculine_words_removed' in improvement:
            st.metric("Bias Words Removed", improvement['masculine_words_removed'])
    with col3:
        if 'inclusive_words_added' in improvement:
            st.metric("Inclusive Terms Added", improvement['inclusive_words_added'])


def display_final_text(rewrite_result):
    """显示最终文本"""
    st.markdown("#### 📋 Final AI-Optimized Job Description")

    # 可编辑的最终文本
    final_text = st.text_area(
        "You can further edit the AI-rewritten text:",
        rewrite_result.rewritten_text,
        height=300,
        help="Feel free to make additional manual adjustments"
    )

    # 操作按钮
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🔍 Re-analyze", help="Analyze your edited version"):
            if final_text.strip() != rewrite_result.rewritten_text:
                new_analysis = analyze_job_description(final_text)
                if new_analysis:
                    st.session_state.current_analysis = new_analysis
                    st.session_state.analysis_history.append(new_analysis)
                    st.success("✅ Re-analysis completed!")
                    st.rerun()

    with col2:
        if st.button("💾 Save Version", help="Save this version"):
            st.success("✅ Version saved!")

    with col3:
        # 导出CSV
        export_data = pd.DataFrame({
            'Original': [rewrite_result.original_text],
            'Rewritten': [final_text],
            'Changes': [len(rewrite_result.changes)]
        })
        csv = export_data.to_csv(index=False)
        st.download_button(
            "📥 Download",
            csv,
            file_name="rewritten_job_description.csv",
            mime="text/csv"
        )

    with col4:
        if st.button("🔄 New Version", help="Generate alternative rewrite"):
            with st.spinner("Generating alternative..."):
                try:
                    rewriter = get_text_rewriter()
                    new_result = rewriter.intelligent_rewrite(rewrite_result.original_text)
                    st.session_state.rewrite_result = new_result
                    st.success("✅ Alternative version generated!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed: {str(e)}")


def main():
    """主应用函数"""
    # 初始化
    initialize_session_state()

    # 显示头部
    display_header()

    # 侧边栏
    with st.sidebar:
        st.header("📝 Analysis Tools")

        # 模式选择
        mode = st.selectbox(
            "Choose Mode",
            ["Single Analysis", "Demo Examples", "Batch Analysis"],
            help="Select how you want to analyze job descriptions"
        )

        # Demo切换
        if mode == "Demo Examples":
            st.session_state.demo_mode = True
            demo_option = st.selectbox(
                "Select Demo",
                ["Biased Example", "Inclusive Example", "Tech Role Example"]
            )
        else:
            st.session_state.demo_mode = False

        # 添加AI改写快捷操作
        if st.session_state.current_analysis:
            st.markdown("---")
            st.header("🤖 AI Rewriter")

            if st.button("✨ Quick Rewrite", help="Instantly rewrite current text"):
                try:
                    rewriter = get_text_rewriter()
                    rewrite_result = rewriter.intelligent_rewrite(st.session_state.current_analysis['text'])
                    st.session_state.rewrite_result = rewrite_result
                    st.session_state.show_rewrite = True
                    st.success("✅ Quick rewrite completed!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed: {str(e)}")

            if st.session_state.rewrite_result:
                improvement = st.session_state.rewrite_result.improvement_prediction
                if 'score_improvement' in improvement:
                    st.metric(
                        "Score Boost",
                        f"+{improvement['score_improvement']:.1f}",
                        help="Predicted score improvement"
                    )

    # 主内容区域
    if st.session_state.demo_mode:
        # Demo模式
        demo_texts = {
            "Biased Example": """We are looking for an aggressive ninja developer who can work independently. 
            The ideal candidate must have strong leadership skills and be highly competitive. 
            This is a demanding role for a rockstar programmer who thrives in a fast-paced environment.""",

            "Inclusive Example": """We welcome a collaborative developer to join our diverse team. 
            We offer flexible work arrangements and support professional development. 
            The ideal candidate will have experience in teamwork and communication.""",

            "Tech Role Example": """Join our supportive engineering team as a software developer. 
            We value collaboration and offer mentorship opportunities. 
            Flexible working hours and career development provided."""
        }

        demo_text = demo_texts[demo_option]
        st.text_area("Demo Job Description", demo_text, height=150, disabled=True)

        if st.button("Analyze Demo", type="primary"):
            analysis_result = analyze_job_description(demo_text)
            if analysis_result:
                st.session_state.current_analysis = analysis_result
                st.session_state.rewrite_result = None  # 清除之前的改写结果
                st.session_state.show_rewrite = False
                st.rerun()

    else:
        # 正常分析模式
        st.subheader("📝 Enter Job Description")

        # 文本输入
        job_text = st.text_area(
            "Paste your job description here:",
            height=200,
            placeholder="Enter the job description you want to analyze for gender bias and inclusivity...",
            help="Paste the complete job description including requirements, responsibilities, and company information."
        )

        # 分析按钮
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            analyze_button = st.button("🔍 Analyze", type="primary", disabled=not job_text.strip())

        with col2:
            if st.button("🗑️ Clear"):
                st.session_state.current_analysis = None
                st.session_state.rewrite_result = None
                st.session_state.show_rewrite = False
                st.rerun()

        # 执行分析
        if analyze_button and job_text.strip():
            # 验证输入
            is_valid, message = validate_text_input(job_text)
            if not is_valid:
                st.error(message)
            else:
                analysis_result = analyze_job_description(job_text)
                if analysis_result:
                    st.session_state.current_analysis = analysis_result
                    st.session_state.analysis_history.append(analysis_result)
                    # 清除之前的改写结果
                    st.session_state.rewrite_result = None
                    st.session_state.show_rewrite = False
                    st.rerun()

    # 显示结果
    if st.session_state.current_analysis:
        st.markdown("---")
        display_metrics_overview()
        st.markdown("---")
        display_detailed_analysis()

    # 页脚
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🚀 <strong>GenderLens AI</strong> - Making hiring more inclusive, one job description at a time</p>
        <p><small>✨ Now with AI-powered rewriting • 🔬 Evidence-based recommendations • 📊 Predictive analytics</small></p>
        <p><small>Built with Streamlit • Powered by Machine Learning • Designed for Equality</small></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()