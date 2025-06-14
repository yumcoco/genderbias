# @Versin:1.0
# @Author:Yummy
"""
常量和配置文件
包含性别偏向词汇、评分权重等核心配置
"""

# 性别偏向词汇库
MASCULINE_WORDS = [
    # 竞争性词汇
    'aggressive', 'assertive', 'competitive', 'dominant', 'forceful',
    'ambitious', 'driven', 'determined', 'decisive', 'confident',

    # 独立性词汇
    'independent', 'individual', 'self-reliant', 'autonomous', 'leader',
    'outspoken', 'strong', 'fearless', 'bold', 'challenging',

    # 科技行业特有
    'ninja', 'rockstar', 'guru', 'wizard', 'champion', 'warrior',
    'hero', 'master', 'expert', 'hacker'
]

FEMININE_WORDS = [
    # 协作性词汇
    'collaborative', 'cooperative', 'supportive', 'understanding',
    'interdependent', 'team', 'together', 'community', 'share',
    'collective', 'joint', 'mutual', 'shared',

    # 关怀性词汇
    'empathetic', 'nurturing', 'considerate', 'caring', 'patient',
    'gentle', 'kind', 'thoughtful', 'compassionate', 'sensitive',

    # 沟通性词汇
    'communicate', 'listen', 'responsive', 'interpersonal', 'relationship'
]

# 包容性关键词
INCLUSIVE_WORDS = [
    'diverse', 'inclusive', 'welcoming', 'flexible', 'balance',
    'opportunity', 'growth', 'development', 'learning', 'support',
    'mentorship', 'training', 'career advancement', 'work-life balance',
    'equal opportunity', 'accommodation', 'accessibility', 'belonging',
    'equity', 'fair', 'open', 'transparent'
]

# 排他性词汇
EXCLUSIVE_WORDS = [
    # 高压力词汇
    'demanding', 'intense', 'fast-paced', 'high-pressure', 'stressful',
    'aggressive deadlines', 'tight deadlines', 'demanding environment',

    # 严格要求词汇
    'must have', 'required', 'essential', 'mandatory', 'critical',
    'absolutely necessary', 'non-negotiable', 'strict requirements',

    # 排他性短语
    'only consider', 'exclusively', 'solely', 'limited to'
]

# 评分权重配置
SCORING_WEIGHTS = {
    'masculine_penalty': -2.0,  # 每个男性化词汇扣分
    'feminine_bonus': 1.5,  # 每个女性化词汇加分
    'inclusive_bonus': 3.0,  # 每个包容性词汇加分
    'exclusive_penalty': -1.5,  # 每个排他性词汇扣分
    'length_factor': 0.01,  # 文本长度因子
    'base_score': 50.0  # 基础分数
}

# 评分等级定义
SCORE_LEVELS = {
    'excellent': {'min': 80, 'color': '#4CAF50', 'label': '优秀'},
    'good': {'min': 65, 'color': '#8BC34A', 'label': '良好'},
    'fair': {'min': 50, 'color': '#FFC107', 'label': '一般'},
    'poor': {'min': 35, 'color': '#FF9800', 'label': '较差'},
    'very_poor': {'min': 0, 'color': '#F44336', 'label': '很差'}
}

# 预测模型特征
FEATURE_NAMES = [
    'masculine_word_count',
    'feminine_word_count',
    'inclusive_word_count',
    'exclusive_word_count',
    'text_length',
    'avg_sentence_length',
    'requirement_intensity',
    'positive_sentiment'
]

# 文本预处理配置
TEXT_PROCESSING = {
    'min_length': 50,  # 最小文本长度
    'max_length': 5000,  # 最大文本长度
    'stop_words': ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'],
    'remove_html': True,  # 是否移除HTML标签
    'lowercase': True  # 是否转换为小写
}

# 改进建议模板
IMPROVEMENT_SUGGESTIONS = {
    'high_masculine': "考虑将竞争性词汇替换为更中性的表达",
    'low_inclusive': "添加更多包容性和多样性相关的词汇",
    'high_exclusive': "减少过于严格或排他性的要求表述",
    'short_text': "职位描述过于简单，建议增加更多细节",
    'long_text': "职位描述过于冗长，建议精简核心要求"
}

# 职位类型分类
JOB_CATEGORIES = [
    'software_engineer', 'data_scientist', 'product_manager',
    'designer', 'analyst', 'consultant', 'manager', 'director',
    'intern', 'senior', 'lead', 'architect'
]

# API配置（如果使用）
API_CONFIG = {
    'max_retries': 3,
    'timeout': 30,
    'rate_limit': 60  # 每分钟请求数
}