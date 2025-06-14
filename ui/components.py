"""
UI组件库
可复用的Streamlit界面组件
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List, Any


def create_sidebar():
    """创建侧边栏"""
    with st.sidebar:
        st.header("🛠️ Tools & Settings")

        # 分析选项
        st.subheader("Analysis Options")

        # 高级设置
        with st.expander("⚙️ Advanced Settings"):
            sensitivity = st.slider(
                "Bias Detection Sensitivity",
                min_value=0.1,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="Higher values detect more subtle bias"
            )

            include_neutral = st.checkbox(
                "Include Neutral Words",
                value=False,
                help="Show words that are neither masculine nor feminine coded"
            )

            detailed_analysis = st.checkbox(
                "Detailed Feature Analysis",
                value=True,
                help="Show detailed breakdown of all features"
            )

        # 快速操作
        st.subheader("Quick Actions")

        if st.button("📋 Copy Analysis"):
            st.info("Analysis copied to clipboard!")

        if st.button("📤 Export Report"):
            st.info("Report exported!")

        if st.button("📊 View History"):
            st.info("Analysis history displayed!")

        return {
            'sensitivity': sensitivity,
            'include_neutral': include_neutral,
            'detailed_analysis': detailed_analysis
        }


def create_score_gauge(score: float, title: str = "Score") -> go.Figure:
    """创建评分仪表盘"""
    # 确定颜色
    if score >= 80:
        color = "#4CAF50"
    elif score >= 65:
        color = "#8BC34A"
    elif score >= 50:
        color = "#FFC107"
    elif score >= 35:
        color = "#FF9800"
    else:
        color = "#F44336"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16}},
        delta={'reference': 50, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 35], 'color': '#ffebee'},
                {'range': [35, 50], 'color': '#fff3e0'},
                {'range': [50, 65], 'color': '#fffde7'},
                {'range': [65, 80], 'color': '#f1f8e9'},
                {'range': [80, 100], 'color': '#e8f5e8'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        font={'color': "darkblue", 'family': "Arial"}
    )

    return fig


def create_word_cloud_chart(words_data: Dict[str, List[str]]) -> go.Figure:
    """创建词汇分布图表"""
    # 准备数据
    categories = []
    words = []
    counts = []
    colors = []

    color_map = {
        'masculine': '#FF5722',
        'feminine': '#9C27B0',
        'inclusive': '#4CAF50',
        'exclusive': '#FF9800'
    }

    for category, word_list in words_data.items():
        if word_list:  # 只显示有词汇的类别
            categories.extend([category.title()] * len(word_list))
            words.extend(word_list)
            counts.extend([1] * len(word_list))  # 每个词计数为1
            colors.extend([color_map.get(category, '#757575')] * len(word_list))

    if not words:
        # 如果没有词汇，创建空图表
        fig = go.Figure()
        fig.add_annotation(
            text="No biased words detected",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16, color="gray")
        )
        fig.update_layout(height=300, title="Word Analysis")
        return fig

    # 创建气泡图
    df = pd.DataFrame({
        'Category': categories,
        'Word': words,
        'Count': counts,
        'Color': colors
    })

    fig = px.scatter(
        df,
        x='Category',
        y='Word',
        size='Count',
        color='Category',
        color_discrete_map={
            'Masculine': '#FF5722',
            'Feminine': '#9C27B0',
            'Inclusive': '#4CAF50',
            'Exclusive': '#FF9800'
        },
        title="Detected Words by Category",
        height=400
    )

    fig.update_traces(marker=dict(sizemin=10, sizemode='diameter'))
    fig.update_layout(
        showlegend=True,
        xaxis_title="Category",
        yaxis_title="Words",
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig


def create_component_scores_chart(component_scores: Dict[str, float]) -> go.Figure:
    """创建组件评分图表"""
    components = list(component_scores.keys())
    scores = list(component_scores.values())

    # 美化组件名称
    component_names = {
        'language_balance': 'Language Balance',
        'inclusivity': 'Inclusivity',
        'openness': 'Openness',
        'text_quality': 'Text Quality',
        'sentiment': 'Sentiment'
    }

    display_names = [component_names.get(comp, comp.title()) for comp in components]

    # 创建条形图
    fig = go.Figure(data=[
        go.Bar(
            x=display_names,
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
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig


def create_prediction_chart(current_rate: float, improved_rate: float = None) -> go.Figure:
    """创建预测对比图表"""
    fig = go.Figure()

    # 当前预测率
    fig.add_trace(go.Bar(
        name='Current Prediction',
        x=['Women Application Rate'],
        y=[current_rate * 100],
        marker_color='#FF9800',
        text=f'{current_rate * 100:.1f}%',
        textposition='auto'
    ))

    # 如果有改进预测，显示对比
    if improved_rate:
        fig.add_trace(go.Bar(
            name='After Improvements',
            x=['Women Application Rate'],
            y=[improved_rate * 100],
            marker_color='#4CAF50',
            text=f'{improved_rate * 100:.1f}%',
            textposition='auto'
        ))

    fig.update_layout(
        title="Predicted Women Application Rate",
        yaxis_title="Percentage (%)",
        yaxis=dict(range=[0, 100]),
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        barmode='group'
    )

    return fig


def display_metric_card(title: str, value: str, delta: str = None, help_text: str = None):
    """显示指标卡片"""
    delta_html = ""
    if delta:
        delta_color = "#4CAF50" if delta.startswith("+") else "#F44336" if delta.startswith("-") else "#757575"
        delta_html = f'<div style="color: {delta_color}; font-size: 0.8rem; margin-top: 4px;">{delta}</div>'

    help_html = ""
    if help_text:
        help_html = f'<div style="color: #666; font-size: 0.7rem; margin-top: 8px;">{help_text}</div>'

    st.markdown(f"""
    <div style="
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    ">
        <div style="font-size: 0.9rem; color: #666; margin-bottom: 4px;">{title}</div>
        <div style="font-size: 1.5rem; font-weight: bold; color: #333;">{value}</div>
        {delta_html}
        {help_html}
    </div>
    """, unsafe_allow_html=True)


def display_suggestion_card(suggestion: str, priority: str = "medium", impact: str = None):
    """显示建议卡片"""
    priority_colors = {
        'high': '#FF5722',
        'medium': '#FF9800',
        'low': '#4CAF50'
    }

    color = priority_colors.get(priority, '#757575')

    impact_html = ""
    if impact:
        impact_html = f'<div style="font-size: 0.8rem; color: #666; margin-top: 8px;">Expected Impact: {impact}</div>'

    st.markdown(f"""
    <div style="
        background: #f9f9f9;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid {color};
        margin: 0.5rem 0;
    ">
        <div style="font-size: 0.9rem; color: #333;">{suggestion}</div>
        <div style="font-size: 0.7rem; color: {color}; margin-top: 4px; font-weight: bold;">Priority: {priority.upper()}</div>
        {impact_html}
    </div>
    """, unsafe_allow_html=True)


def display_word_highlights(text: str, words_to_highlight: Dict[str, List[str]]) -> str:
    """高亮显示文本中的关键词"""
    highlighted_text = text

    # 定义高亮颜色
    highlight_colors = {
        'masculine': '#ffcdd2',
        'feminine': '#e1bee7',
        'inclusive': '#c8e6c9',
        'exclusive': '#ffe0b2'
    }

    # 对每个类别的词汇进行高亮
    for category, words in words_to_highlight.items():
        color = highlight_colors.get(category, '#f5f5f5')
        for word in words:
            if word.lower() in highlighted_text.lower():
                # 使用正则表达式进行大小写不敏感的替换
                import re
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                highlighted_text = pattern.sub(
                    f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px; font-weight: bold;">{word}</span>',
                    highlighted_text
                )

    return highlighted_text


def create_comparison_table(analyses: List[Dict]) -> pd.DataFrame:
    """创建分析结果对比表"""
    if not analyses:
        return pd.DataFrame()

    comparison_data = []

    for i, analysis in enumerate(analyses):
        row = {
            'Version': f'Version {i + 1}',
            'Inclusivity Score': f"{analysis['inclusivity_score'].overall_score:.1f}",
            'Grade': analysis['inclusivity_score'].grade,
            'Predicted Women %': f"{analysis['prediction']['percentage']:.1f}%",
            'Bias Direction': analysis['bias_analysis'].overall_bias.title(),
            'Masculine Words': len(analysis['bias_analysis'].masculine_words),
            'Inclusive Words': len(analysis['bias_analysis'].inclusive_words),
            'Recommendations': len(analysis['inclusivity_score'].recommendations)
        }
        comparison_data.append(row)

    return pd.DataFrame(comparison_data)


def display_progress_indicator(current_step: int, total_steps: int, step_names: List[str]):
    """显示进度指示器"""
    progress_html = '<div style="display: flex; align-items: center; margin: 1rem 0;">'

    for i in range(total_steps):
        # 确定步骤状态
        if i < current_step:
            color = "#4CAF50"
            icon = "✅"
        elif i == current_step:
            color = "#2196F3"
            icon = "🔄"
        else:
            color = "#E0E0E0"
            icon = "⭕"

        step_name = step_names[i] if i < len(step_names) else f"Step {i + 1}"

        progress_html += f"""
        <div style="
            flex: 1;
            text-align: center;
            padding: 0.5rem;
            margin: 0 0.2rem;
            background-color: {color};
            color: white;
            border-radius: 4px;
            font-size: 0.8rem;
        ">
            {icon} {step_name}
        </div>
        """

    progress_html += '</div>'
    st.markdown(progress_html, unsafe_allow_html=True)


def create_export_options():
    """创建导出选项"""
    st.subheader("📤 Export Options")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📋 Copy to Clipboard", help="Copy analysis results to clipboard"):
            st.success("Analysis copied to clipboard!")

    with col2:
        if st.button("📄 Download PDF", help="Download analysis as PDF report"):
            st.info("PDF download feature coming soon!")

    with col3:
        if st.button("📊 Export CSV", help="Export data as CSV file"):
            st.info("CSV export feature coming soon!")


def display_analysis_history():
    """显示分析历史"""
    if 'analysis_history' not in st.session_state or not st.session_state.analysis_history:
        st.info("No analysis history available")
        return

    st.subheader("📚 Analysis History")

    # 创建历史记录表
    history_data = []
    for i, analysis in enumerate(st.session_state.analysis_history):
        history_data.append({
            'ID': i + 1,
            'Time': analysis['timestamp'].strftime('%H:%M:%S'),
            'Score': f"{analysis['inclusivity_score'].overall_score:.1f}",
            'Grade': analysis['inclusivity_score'].grade,
            'Bias': analysis['bias_analysis'].overall_bias.title(),
            'Text Preview': analysis['text'][:50] + "..." if len(analysis['text']) > 50 else analysis['text']
        })

    history_df = pd.DataFrame(history_data)

    # 显示表格
    selected_row = st.selectbox(
        "Select analysis to view:",
        options=range(len(history_df)),
        format_func=lambda x: f"#{x + 1} - {history_df.iloc[x]['Time']} - Score: {history_df.iloc[x]['Score']}"
    )

    if st.button("Load Selected Analysis"):
        st.session_state.current_analysis = st.session_state.analysis_history[selected_row]
        st.success(f"Loaded analysis #{selected_row + 1}")
        st.rerun()


def create_feature_importance_chart(feature_importance: Dict[str, float]) -> go.Figure:
    """创建特征重要性图表"""
    if not feature_importance:
        fig = go.Figure()
        fig.add_annotation(
            text="Feature importance not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16, color="gray")
        )
        fig.update_layout(height=300, title="Feature Importance")
        return fig

    features = list(feature_importance.keys())
    importance = list(feature_importance.values())

    # 美化特征名称
    feature_names = {
        'masculine_word_count': 'Masculine Words',
        'feminine_word_count': 'Feminine Words',
        'inclusive_word_count': 'Inclusive Words',
        'exclusive_word_count': 'Exclusive Words',
        'text_length': 'Text Length',
        'avg_sentence_length': 'Avg Sentence Length',
        'sentiment_score': 'Sentiment',
        'masculine_density': 'Masculine Density',
        'inclusive_density': 'Inclusive Density'
    }

    display_features = [feature_names.get(f, f.replace('_', ' ').title()) for f in features]

    # 创建水平条形图
    fig = go.Figure(data=[
        go.Bar(
            x=importance,
            y=display_features,
            orientation='h',
            marker_color='#667eea',
            text=[f'{imp:.3f}' for imp in importance],
            textposition='auto',
        )
    ])

    fig.update_layout(
        title="Feature Importance in Prediction Model",
        xaxis_title="Importance Score",
        yaxis_title="Features",
        height=400,
        margin=dict(l=120, r=20, t=40, b=20)
    )

    return fig


def display_tooltip_info(text: str, tooltip: str):
    """显示带提示信息的文本"""
    st.markdown(f"""
    <div title="{tooltip}" style="cursor: help; border-bottom: 1px dotted #666;">
        {text}
    </div>
    """, unsafe_allow_html=True)


def create_demo_showcase():
    """创建演示展示区域"""
    st.subheader("🎯 Demo Examples")

    demo_examples = {
        "Highly Biased (Masculine)": {
            "text": "We need an aggressive ninja developer who must dominate the competition. This demanding role requires a rockstar who can work independently and crush deadlines.",
            "expected_score": "Low (20-40)",
            "bias_type": "Masculine"
        },
        "Well Balanced": {
            "text": "We welcome a collaborative developer to join our diverse team. We offer flexible work arrangements and support professional development.",
            "expected_score": "Good (70-85)",
            "bias_type": "Neutral"
        },
        "Highly Inclusive": {
            "text": "Join our supportive and inclusive engineering team. We value diversity, offer mentorship programs, and promote work-life balance. All backgrounds welcome.",
            "expected_score": "Excellent (85-95)",
            "bias_type": "Inclusive"
        }
    }

    for title, example in demo_examples.items():
        with st.expander(f"📝 {title}"):
            st.write(f"**Expected Score:** {example['expected_score']}")
            st.write(f"**Bias Type:** {example['bias_type']}")
            st.text_area("Job Description:", example['text'], height=100, disabled=True, key=f"demo_{title}")
            if st.button(f"Analyze {title}", key=f"btn_{title}"):
                # 这里可以触发分析
                st.info(f"Analyzing {title}...")


def display_results(analysis_result: Dict[str, Any]):
    """显示分析结果的统一接口"""
    if not analysis_result:
        st.warning("No analysis results to display")
        return

    # 创建结果展示区域
    with st.container():
        # 快速概览
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            display_metric_card(
                "Inclusivity Score",
                f"{analysis_result['inclusivity_score'].overall_score:.1f}/100",
                help_text=analysis_result['inclusivity_score'].grade
            )

        with col2:
            display_metric_card(
                "Predicted Women Applicants",
                f"{analysis_result['prediction']['percentage']:.1f}%",
                help_text="Expected application rate"
            )

        with col3:
            bias_direction = analysis_result['bias_analysis'].overall_bias.title()
            display_metric_card(
                "Bias Direction",
                bias_direction,
                help_text=f"Strength: {analysis_result['bias_analysis'].bias_strength:.2f}"
            )

        with col4:
            recommendations_count = len(analysis_result['inclusivity_score'].recommendations)
            display_metric_card(
                "Recommendations",
                str(recommendations_count),
                help_text="Improvement suggestions"
            )

        # 详细图表
        st.markdown("---")

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            # 评分仪表盘
            gauge_fig = create_score_gauge(
                analysis_result['inclusivity_score'].overall_score,
                "Inclusivity Score"
            )
            st.plotly_chart(gauge_fig, use_container_width=True)

        with chart_col2:
            # 组件评分
            component_fig = create_component_scores_chart(
                analysis_result['inclusivity_score'].component_scores
            )
            st.plotly_chart(component_fig, use_container_width=True)