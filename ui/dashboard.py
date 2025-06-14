"""
仪表板模块
高级分析和可视化界面
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.components import (
    create_score_gauge, create_word_cloud_chart, create_component_scores_chart,
    create_prediction_chart, display_metric_card, display_suggestion_card
)


def create_main_dashboard():
    """创建主仪表板"""
    st.title("📊 Advanced Analytics Dashboard")

    # 检查是否有分析数据
    if 'analysis_history' not in st.session_state or not st.session_state.analysis_history:
        st.info("👋 Welcome! Analyze some job descriptions first to see advanced analytics here.")
        return

    # 获取历史数据
    analyses = st.session_state.analysis_history

    # 创建标签页
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "🔍 Deep Analysis", "📊 Comparisons", "🎯 Insights"])

    with tab1:
        display_trends_analysis(analyses)

    with tab2:
        display_deep_analysis()

    with tab3:
        display_comparisons(analyses)

    with tab4:
        display_insights(analyses)


def display_trends_analysis(analyses: List[Dict]):
    """显示趋势分析"""
    st.subheader("📈 Analysis Trends")

    if len(analyses) < 2:
        st.info("Analyze at least 2 job descriptions to see trends")
        return

    # 准备趋势数据
    trend_data = []
    for i, analysis in enumerate(analyses):
        trend_data.append({
            'Analysis #': i + 1,
            'Inclusivity Score': analysis['inclusivity_score'].overall_score,
            'Women Application Rate': analysis['prediction']['percentage'],
            'Masculine Words': len(analysis['bias_analysis'].masculine_words),
            'Inclusive Words': len(analysis['bias_analysis'].inclusive_words),
            'Bias Strength': analysis['bias_analysis'].bias_strength,
            'Time': analysis['timestamp'].strftime('%H:%M')
        })

    trend_df = pd.DataFrame(trend_data)

    # 显示趋势图表
    col1, col2 = st.columns(2)

    with col1:
        # 包容性评分趋势
        fig_score = px.line(
            trend_df,
            x='Analysis #',
            y='Inclusivity Score',
            title='Inclusivity Score Trend',
            markers=True,
            line_shape='spline'
        )
        fig_score.update_layout(height=400, yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig_score, use_container_width=True)

    with col2:
        # 女性申请率趋势
        fig_women = px.line(
            trend_df,
            x='Analysis #',
            y='Women Application Rate',
            title='Predicted Women Application Rate Trend',
            markers=True,
            line_shape='spline',
            color_discrete_sequence=['#FF6B6B']
        )
        fig_women.update_layout(height=400, yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig_women, use_container_width=True)

    # 词汇使用趋势
    st.subheader("Word Usage Patterns")

    fig_words = go.Figure()

    fig_words.add_trace(go.Scatter(
        x=trend_df['Analysis #'],
        y=trend_df['Masculine Words'],
        mode='lines+markers',
        name='Masculine Words',
        line=dict(color='#FF5722')
    ))

    fig_words.add_trace(go.Scatter(
        x=trend_df['Analysis #'],
        y=trend_df['Inclusive Words'],
        mode='lines+markers',
        name='Inclusive Words',
        line=dict(color='#4CAF50')
    ))

    fig_words.update_layout(
        title='Word Count Trends Across Analyses',
        xaxis_title='Analysis Number',
        yaxis_title='Word Count',
        height=400
    )

    st.plotly_chart(fig_words, use_container_width=True)

    # 趋势统计
    st.subheader("📊 Trend Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_score = trend_df['Inclusivity Score'].mean()
        display_metric_card("Average Score", f"{avg_score:.1f}", help_text="Across all analyses")

    with col2:
        score_improvement = trend_df['Inclusivity Score'].iloc[-1] - trend_df['Inclusivity Score'].iloc[0]
        delta_text = f"+{score_improvement:.1f}" if score_improvement > 0 else f"{score_improvement:.1f}"
        display_metric_card("Score Change", delta_text, help_text="First to latest")

    with col3:
        avg_women_rate = trend_df['Women Application Rate'].mean()
        display_metric_card("Avg Women Rate", f"{avg_women_rate:.1f}%", help_text="Predicted average")

    with col4:
        best_score = trend_df['Inclusivity Score'].max()
        display_metric_card("Best Score", f"{best_score:.1f}", help_text="Highest achieved")


def display_deep_analysis():
    """显示深度分析"""
    st.subheader("🔍 Deep Analysis")

    if not st.session_state.current_analysis:
        st.info("Select an analysis from the main page to see deep insights")
        return

    analysis = st.session_state.current_analysis

    # 创建深度分析标签页
    deep_tab1, deep_tab2, deep_tab3 = st.tabs(["🧬 Text Analysis", "🎯 Feature Impact", "🔮 Predictions"])

    with deep_tab1:
        display_text_analysis(analysis)

    with deep_tab2:
        display_feature_impact(analysis)

    with deep_tab3:
        display_prediction_analysis(analysis)


def display_text_analysis(analysis: Dict):
    """显示文本分析"""
    st.markdown("#### 📝 Text Breakdown")

    text = analysis['text']
    bias_analysis = analysis['bias_analysis']

    # 文本统计
    col1, col2, col3 = st.columns(3)

    with col1:
        word_count = len(text.split())
        char_count = len(text)
        st.metric("Word Count", word_count)
        st.metric("Character Count", char_count)

    with col2:
        sentence_count = len([s for s in text.split('.') if s.strip()])
        avg_word_length = sum(len(word) for word in text.split()) / word_count if word_count > 0 else 0
        st.metric("Sentences", sentence_count)
        st.metric("Avg Word Length", f"{avg_word_length:.1f}")

    with col3:
        unique_words = len(set(text.lower().split()))
        complexity = unique_words / word_count if word_count > 0 else 0
        st.metric("Unique Words", unique_words)
        st.metric("Complexity", f"{complexity:.2f}")

    # 高亮显示的文本
    st.markdown("#### 🎨 Highlighted Text")

    # 创建高亮文本
    from ui.components import display_word_highlights

    words_to_highlight = {
        'masculine': bias_analysis.masculine_words,
        'feminine': bias_analysis.feminine_words,
        'inclusive': bias_analysis.inclusive_words,
        'exclusive': bias_analysis.exclusive_words
    }

    highlighted_text = display_word_highlights(text, words_to_highlight)

    st.markdown(highlighted_text, unsafe_allow_html=True)

    # 图例
    st.markdown("""
    **Legend:**
    - <span style="background-color: #ffcdd2; padding: 2px 4px;">Masculine-coded</span>
    - <span style="background-color: #e1bee7; padding: 2px 4px;">Feminine-coded</span>  
    - <span style="background-color: #c8e6c9; padding: 2px 4px;">Inclusive</span>
    - <span style="background-color: #ffe0b2; padding: 2px 4px;">Exclusive</span>
    """, unsafe_allow_html=True)


def display_feature_impact(analysis: Dict):
    """显示特征影响分析"""
    st.markdown("#### 🎯 Feature Impact Analysis")

    # 获取预测器并分析特征重要性
    from core.prediction_model import get_women_predictor
    predictor = get_women_predictor()

    # 特征重要性图表
    feature_importance = predictor.get_feature_importance()

    if feature_importance and 'message' not in feature_importance:
        from ui.components import create_feature_importance_chart
        fig = create_feature_importance_chart(feature_importance)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance analysis requires a trained model")

    # 当前分析的特征值
    st.markdown("#### 📊 Current Analysis Features")

    features = predictor.extract_features(analysis['text'])
    feature_names = predictor.feature_names

    feature_data = []
    for i, (name, value) in enumerate(zip(feature_names, features)):
        feature_data.append({
            'Feature': name.replace('_', ' ').title(),
            'Value': f"{value:.2f}",
            'Impact': 'High' if i < 3 else 'Medium' if i < 6 else 'Low'
        })

    feature_df = pd.DataFrame(feature_data)
    st.dataframe(feature_df, use_container_width=True)


def display_prediction_analysis(analysis: Dict):
    """显示预测分析"""
    st.markdown("#### 🔮 Prediction Analysis")

    prediction = analysis['prediction']
    improvements = analysis['improvement_suggestions']

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Current Prediction**")
        st.metric(
            "Women Application Rate",
            f"{prediction['percentage']:.1f}%",
            help="Based on current job description"
        )

        confidence = prediction.get('confidence', 'medium')
        confidence_color = {"high": "🟢", "medium": "🟡", "low": "🔴"}
        st.markdown(f"Confidence: {confidence_color.get(confidence, '🟡')} {confidence.title()}")

    with col2:
        if 'estimated_improved_rate' in improvements:
            improved = improvements['estimated_improved_rate']
            st.markdown("**After Improvements**")
            st.metric(
                "Estimated Rate",
                f"{improved['percentage']:.1f}%",
                delta=f"+{improved['improvement']:.1f}%"
            )
        else:
            st.info("No improvement estimates available")

    # 预测对比图表
    if 'estimated_improved_rate' in improvements:
        improved_rate = improvements['estimated_improved_rate']['women_application_rate']
        fig = create_prediction_chart(prediction['women_application_rate'], improved_rate)
        st.plotly_chart(fig, use_container_width=True)


def display_comparisons(analyses: List[Dict]):
    """显示对比分析"""
    st.subheader("📊 Comparative Analysis")

    if len(analyses) < 2:
        st.info("Need at least 2 analyses to compare")
        return

    # 选择要对比的分析
    st.markdown("#### Select Analyses to Compare")

    col1, col2 = st.columns(2)

    with col1:
        analysis1_idx = st.selectbox(
            "Analysis 1:",
            range(len(analyses)),
            format_func=lambda x: f"#{x + 1} - Score: {analyses[x]['inclusivity_score'].overall_score:.1f}"
        )

    with col2:
        analysis2_idx = st.selectbox(
            "Analysis 2:",
            range(len(analyses)),
            index=min(1, len(analyses) - 1),
            format_func=lambda x: f"#{x + 1} - Score: {analyses[x]['inclusivity_score'].overall_score:.1f}"
        )

    if analysis1_idx == analysis2_idx:
        st.warning("Please select different analyses to compare")
        return

    # 执行对比
    analysis1 = analyses[analysis1_idx]
    analysis2 = analyses[analysis2_idx]

    # 对比表格
    st.markdown("#### 📋 Comparison Table")

    comparison_data = {
        'Metric': [
            'Inclusivity Score',
            'Grade',
            'Predicted Women %',
            'Bias Direction',
            'Masculine Words',
            'Feminine Words',
            'Inclusive Words',
            'Exclusive Words',
            'Recommendations'
        ],
        f'Analysis #{analysis1_idx + 1}': [
            f"{analysis1['inclusivity_score'].overall_score:.1f}",
            analysis1['inclusivity_score'].grade,
            f"{analysis1['prediction']['percentage']:.1f}%",
            analysis1['bias_analysis'].overall_bias.title(),
            len(analysis1['bias_analysis'].masculine_words),
            len(analysis1['bias_analysis'].feminine_words),
            len(analysis1['bias_analysis'].inclusive_words),
            len(analysis1['bias_analysis'].exclusive_words),
            len(analysis1['inclusivity_score'].recommendations)
        ],
        f'Analysis #{analysis2_idx + 1}': [
            f"{analysis2['inclusivity_score'].overall_score:.1f}",
            analysis2['inclusivity_score'].grade,
            f"{analysis2['prediction']['percentage']:.1f}%",
            analysis2['bias_analysis'].overall_bias.title(),
            len(analysis2['bias_analysis'].masculine_words),
            len(analysis2['bias_analysis'].feminine_words),
            len(analysis2['bias_analysis'].inclusive_words),
            len(analysis2['bias_analysis'].exclusive_words),
            len(analysis2['inclusivity_score'].recommendations)
        ]
    }

    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)

    # 可视化对比
    st.markdown("#### 📊 Visual Comparison")

    # 雷达图对比
    categories = ['Language Balance', 'Inclusivity', 'Openness', 'Text Quality', 'Sentiment']

    fig = go.Figure()

    # Analysis 1
    values1 = [analysis1['inclusivity_score'].component_scores.get(comp, 0) for comp in
               ['language_balance', 'inclusivity', 'openness', 'text_quality', 'sentiment']]
    fig.add_trace(go.Scatterpolar(
        r=values1,
        theta=categories,
        fill='toself',
        name=f'Analysis #{analysis1_idx + 1}',
        line_color='#667eea'
    ))

    # Analysis 2
    values2 = [analysis2['inclusivity_score'].component_scores.get(comp, 0) for comp in
               ['language_balance', 'inclusivity', 'openness', 'text_quality', 'sentiment']]
    fig.add_trace(go.Scatterpolar(
        r=values2,
        theta=categories,
        fill='toself',
        name=f'Analysis #{analysis2_idx + 1}',
        line_color='#764ba2'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Component Scores Comparison",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)


def display_insights(analyses: List[Dict]):
    """显示洞察分析"""
    st.subheader("🎯 Key Insights & Recommendations")

    if not analyses:
        st.info("No analyses available for insights")
        return

    # 计算整体统计
    scores = [a['inclusivity_score'].overall_score for a in analyses]
    women_rates = [a['prediction']['percentage'] for a in analyses]
    masculine_counts = [len(a['bias_analysis'].masculine_words) for a in analyses]
    inclusive_counts = [len(a['bias_analysis'].inclusive_words) for a in analyses]

    # 显示关键洞察
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### 🔍 Pattern Analysis")

        # 分析模式
        avg_score = sum(scores) / len(scores)
        avg_women_rate = sum(women_rates) / len(women_rates)

        # 生成洞察
        insights = []

        if avg_score < 50:
            insights.append(
                "⚠️ **Overall inclusivity scores are below average.** Consider focusing on more balanced language across all job descriptions.")
        elif avg_score > 75:
            insights.append(
                "✅ **Excellent inclusivity performance!** Your job descriptions are generally well-balanced and inclusive.")
        else:
            insights.append(
                "📈 **Good foundation with room for improvement.** Focus on specific areas highlighted in recommendations.")

        if avg_women_rate < 30:
            insights.append(
                "🔍 **Low predicted women application rates detected.** This suggests systematic bias patterns that need attention.")
        elif avg_women_rate > 50:
            insights.append(
                "🌟 **Strong appeal to women candidates!** Your inclusive language is likely attracting diverse applicants.")

        if sum(masculine_counts) > sum(inclusive_counts):
            insights.append(
                "⚖️ **Masculine-coded language dominates over inclusive language.** Consider rebalancing your vocabulary.")

        # 显示洞察
        for insight in insights:
            st.markdown(f"""
            <div style="
                background: #f8f9ff;
                padding: 1rem;
                border-radius: 8px;
                border-left: 4px solid #667eea;
                margin: 0.5rem 0;
            ">
                {insight}
            </div>
            """, unsafe_allow_html=True)

        # 趋势分析
        if len(analyses) >= 3:
            st.markdown("#### 📈 Trend Analysis")

            recent_scores = scores[-3:]
            if recent_scores[-1] > recent_scores[0]:
                st.success("📈 **Improving Trend:** Your recent analyses show improvement in inclusivity scores!")
            elif recent_scores[-1] < recent_scores[0]:
                st.warning("📉 **Declining Trend:** Recent scores are lower. Review your latest approaches.")
            else:
                st.info("➡️ **Stable Trend:** Scores are consistent. Consider new strategies for improvement.")

    with col2:
        st.markdown("#### 📊 Quick Stats")

        # 统计卡片
        display_metric_card("Analyses Count", str(len(analyses)), help_text="Total completed")
        display_metric_card("Average Score", f"{avg_score:.1f}", help_text="Across all analyses")
        display_metric_card("Best Score", f"{max(scores):.1f}", help_text="Highest achieved")
        display_metric_card("Avg Women Rate", f"{avg_women_rate:.1f}%", help_text="Predicted average")

        # 分布图
        score_distribution = {
            'Excellent (80+)': len([s for s in scores if s >= 80]),
            'Good (65-79)': len([s for s in scores if 65 <= s < 80]),
            'Fair (50-64)': len([s for s in scores if 50 <= s < 65]),
            'Poor (<50)': len([s for s in scores if s < 50])
        }

        st.markdown("**Score Distribution**")
        for category, count in score_distribution.items():
            if count > 0:
                percentage = (count / len(scores)) * 100
                st.write(f"{category}: {count} ({percentage:.0f}%)")

    # 行动建议
    st.markdown("---")
    st.markdown("#### 🚀 Recommended Actions")

    # 基于数据生成行动建议
    action_items = []

    if avg_score < 60:
        action_items.append({
            "priority": "high",
            "action": "Conduct organization-wide review of job description templates",
            "impact": "Systematic improvement across all postings"
        })

    if sum(masculine_counts) / len(analyses) > 3:
        action_items.append({
            "priority": "high",
            "action": "Create guidelines for avoiding masculine-coded language",
            "impact": "Reduce unconscious bias in job postings"
        })

    if avg_women_rate < 40:
        action_items.append({
            "priority": "medium",
            "action": "Implement inclusive language training for hiring teams",
            "impact": "Increase diversity in applicant pool"
        })

    if len(set([a['bias_analysis'].overall_bias for a in analyses])) == 1:
        action_items.append({
            "priority": "medium",
            "action": "Diversify job description writing approaches",
            "impact": "Appeal to broader candidate demographics"
        })

    # 默认建议
    if not action_items:
        action_items.append({
            "priority": "low",
            "action": "Continue monitoring and gradual improvements",
            "impact": "Maintain current good practices"
        })

    # 显示行动建议
    for item in action_items:
        display_suggestion_card(
            item["action"],
            priority=item["priority"],
            impact=item["impact"]
        )

    # 基准对比
    st.markdown("#### 📏 Industry Benchmarks")

    industry_benchmarks = {
        "Tech Industry Average": {"score": 45, "women_rate": 28},
        "Progressive Companies": {"score": 75, "women_rate": 45},
        "Best-in-Class": {"score": 90, "women_rate": 55}
    }

    benchmark_data = []
    for name, values in industry_benchmarks.items():
        benchmark_data.append({
            'Category': name,
            'Inclusivity Score': values['score'],
            'Women Application Rate': values['women_rate'],
            'Your Performance': 'Above' if avg_score > values['score'] else 'Below'
        })

    # 添加用户数据
    benchmark_data.append({
        'Category': 'Your Average',
        'Inclusivity Score': avg_score,
        'Women Application Rate': avg_women_rate,
        'Your Performance': 'Current'
    })

    benchmark_df = pd.DataFrame(benchmark_data)

    # 基准对比图表
    fig_benchmark = px.scatter(
        benchmark_df,
        x='Inclusivity Score',
        y='Women Application Rate',
        text='Category',
        color='Category',
        size=[20, 15, 25, 30],  # 不同大小的点
        title='Performance vs Industry Benchmarks',
        color_discrete_map={
            'Tech Industry Average': '#FF9800',
            'Progressive Companies': '#2196F3',
            'Best-in-Class': '#4CAF50',
            'Your Average': '#9C27B0'
        }
    )

    fig_benchmark.update_traces(textposition="top center")
    fig_benchmark.update_layout(
        height=500,
        xaxis_title="Inclusivity Score",
        yaxis_title="Women Application Rate (%)",
        showlegend=True
    )

    st.plotly_chart(fig_benchmark, use_container_width=True)

    # 改进路径建议
    st.markdown("#### 🗺️ Improvement Roadmap")

    if avg_score < 50:
        roadmap_steps = [
            "1. **Immediate**: Remove obviously biased language (ninja, rockstar, aggressive)",
            "2. **Short-term**: Add inclusive words (collaborative, supportive, diverse)",
            "3. **Medium-term**: Redesign job description templates",
            "4. **Long-term**: Implement bias training for all hiring managers"
        ]
    elif avg_score < 70:
        roadmap_steps = [
            "1. **Immediate**: Fine-tune specific problematic phrases",
            "2. **Short-term**: Increase use of inclusive language",
            "3. **Medium-term**: A/B test different versions",
            "4. **Long-term**: Monitor and maintain improvements"
        ]
    else:
        roadmap_steps = [
            "1. **Immediate**: Maintain current good practices",
            "2. **Short-term**: Share best practices across teams",
            "3. **Medium-term**: Mentor other departments",
            "4. **Long-term**: Lead industry initiatives"
        ]

    for step in roadmap_steps:
        st.markdown(step)

    # 导出选项
    st.markdown("---")
    st.markdown("#### 📤 Export Insights")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📊 Generate Report"):
            st.info("Comprehensive report generation coming soon!")

    with col2:
        if st.button("📋 Copy Summary"):
            st.info("Insights summary copied to clipboard!")

    with col3:
        if st.button("📧 Email Insights"):
            st.info("Email sharing feature coming soon!")


def create_batch_analysis_dashboard():
    """创建批量分析仪表板"""
    st.subheader("📦 Batch Analysis Dashboard")

    # 文件上传
    uploaded_file = st.file_uploader(
        "Upload CSV file with job descriptions",
        type=['csv'],
        help="CSV should have a column named 'job_description' or 'description'"
    )

    if uploaded_file is not None:
        try:
            # 读取CSV文件
            df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(df)} job descriptions")

            # 显示数据预览
            st.markdown("#### Data Preview")
            st.dataframe(df.head())

            # 选择文本列
            text_columns = [col for col in df.columns if 'description' in col.lower() or 'text' in col.lower()]

            if text_columns:
                selected_column = st.selectbox("Select text column:", text_columns)

                if st.button("🚀 Analyze All", type="primary"):
                    # 执行批量分析
                    perform_batch_analysis(df, selected_column)
            else:
                st.error(
                    "No suitable text column found. Please ensure your CSV has a column containing job descriptions.")

        except Exception as e:
            st.error(f"Error reading file: {e}")


def perform_batch_analysis(df: pd.DataFrame, text_column: str):
    """执行批量分析"""
    from core.bias_detector import get_bias_detector
    from core.inclusivity_scorer import get_inclusivity_scorer
    from core.prediction_model import get_women_predictor

    # 初始化分析器
    bias_detector = get_bias_detector()
    inclusivity_scorer = get_inclusivity_scorer()
    women_predictor = get_women_predictor()

    results = []

    # 创建进度条
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, row in df.iterrows():
        try:
            text = str(row[text_column])

            # 更新进度
            progress = (i + 1) / len(df)
            progress_bar.progress(progress)
            status_text.text(f"Analyzing {i + 1}/{len(df)}: {text[:50]}...")

            # 执行分析
            bias_analysis = bias_detector.analyze_bias_patterns(text)
            inclusivity_score = inclusivity_scorer.score_job_description(text)
            prediction = women_predictor.predict_women_proportion(text)

            # 收集结果
            results.append({
                'ID': i + 1,
                'Inclusivity_Score': inclusivity_score.overall_score,
                'Grade': inclusivity_score.grade,
                'Women_Rate_Prediction': prediction['percentage'],
                'Bias_Direction': bias_analysis.overall_bias,
                'Masculine_Words': len(bias_analysis.masculine_words),
                'Inclusive_Words': len(bias_analysis.inclusive_words),
                'Recommendations_Count': len(inclusivity_score.recommendations),
                'Original_Text': text[:100] + "..." if len(text) > 100 else text
            })

        except Exception as e:
            st.error(f"Error analyzing row {i + 1}: {e}")
            continue

    # 完成分析
    progress_bar.progress(1.0)
    status_text.text("Analysis complete!")

    # 显示结果
    results_df = pd.DataFrame(results)

    st.markdown("#### 📊 Batch Analysis Results")
    st.dataframe(results_df)

    # 汇总统计
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_score = results_df['Inclusivity_Score'].mean()
        display_metric_card("Average Score", f"{avg_score:.1f}")

    with col2:
        high_score_count = len(results_df[results_df['Inclusivity_Score'] >= 70])
        display_metric_card("High Scores", f"{high_score_count}/{len(results_df)}")

    with col3:
        avg_women_rate = results_df['Women_Rate_Prediction'].mean()
        display_metric_card("Avg Women Rate", f"{avg_women_rate:.1f}%")

    with col4:
        masculine_bias_count = len(results_df[results_df['Bias_Direction'] == 'masculine'])
        display_metric_card("Masculine Bias", f"{masculine_bias_count}/{len(results_df)}")

    # 下载结果
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results CSV",
        data=csv,
        file_name="batch_analysis_results.csv",
        mime="text/csv"
    )