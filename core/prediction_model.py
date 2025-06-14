"""
女性申请率预测模型
基于职位描述预测女性申请者比例，输出英文结果
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import get_text_statistics, calculate_sentiment
from core.bias_detector import get_bias_detector
from data.data_loader import get_data_loader


class WomenApplicationPredictor:
    """女性申请率预测器主类"""

    def __init__(self):
        """初始化预测器，设置特征和模型"""
        self.bias_detector = get_bias_detector()
        self.data_loader = get_data_loader()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'masculine_word_count',
            'feminine_word_count',
            'inclusive_word_count',
            'exclusive_word_count',
            'text_length',
            'avg_sentence_length',
            'sentiment_score',
            'masculine_density',
            'inclusive_density'
        ]
        self.is_trained = False
        self.model_performance = {}

        print("Women Application Predictor initialized")

    def extract_features(self, text: str) -> np.ndarray:
        """从文本中提取机器学习特征向量"""
        # 获取偏向分析结果
        bias_analysis = self.bias_detector.analyze_bias_patterns(text)

        # 获取文本统计信息
        text_stats = get_text_statistics(text)

        # 获取情感评分
        sentiment = calculate_sentiment(text)

        # 构建特征向量
        features = [
            len(bias_analysis.masculine_words),  # 男性化词汇数量
            len(bias_analysis.feminine_words),  # 女性化词汇数量
            len(bias_analysis.inclusive_words),  # 包容性词汇数量
            len(bias_analysis.exclusive_words),  # 排他性词汇数量
            text_stats['word_count'],  # 文本长度
            text_stats['avg_sentence_length'],  # 平均句子长度
            sentiment,  # 情感评分
            # 密度特征
            len(bias_analysis.masculine_words) / max(text_stats['word_count'], 1) * 100,  # 男性化词汇密度
            len(bias_analysis.inclusive_words) / max(text_stats['word_count'], 1) * 100,  # 包容性词汇密度
        ]

        return np.array(features)

    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """准备训练数据，提取特征和标签"""
        print("Preparing training data...")

        # 加载数据集
        if not self.data_loader.load_all_datasets():
            raise Exception("Failed to load datasets")

        # 获取合并的训练数据
        training_df = self.data_loader.get_combined_training_data()
        if training_df is None or len(training_df) == 0:
            raise Exception("Training data is empty")

        print(f"Training data loaded: {len(training_df)} samples")

        # 提取特征和标签
        X = []
        y = []

        for idx, row in training_df.iterrows():
            try:
                text = row['description']
                women_prop = row['women_proportion']

                # 跳过无效数据
                if pd.isna(women_prop) or not isinstance(text, str) or len(text.strip()) < 20:
                    continue

                # 提取特征
                features = self.extract_features(text)
                X.append(features)
                y.append(women_prop)

                if len(X) % 100 == 0:
                    print(f"   Processed {len(X)} samples...")

            except Exception as e:
                print(f"   Error processing sample: {e}")
                continue

        X = np.array(X)
        y = np.array(y)

        print(f"Feature extraction completed: {X.shape[0]} valid samples, {X.shape[1]} features")
        print(f"Women application rate distribution: mean={y.mean():.3f}, range=[{y.min():.3f}, {y.max():.3f}]")

        return X, y

    def train_model(self, test_size: float = 0.2) -> Dict:
        """训练预测模型并评估性能"""
        print("Starting model training...")

        # 准备数据
        X, y = self.prepare_training_data()

        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)

        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )

        # 训练多个模型并选择最佳
        models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                min_samples_split=5,
                min_samples_leaf=2
            ),
            'LinearRegression': LinearRegression()
        }

        best_model = None
        best_score = -float('inf')
        results = {}

        for name, model in models.items():
            print(f"   Training {name}...")

            # 训练模型
            model.fit(X_train, y_train)

            # 预测
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            # 评估
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            test_mae = mean_absolute_error(y_test, test_pred)

            # 交叉验证
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

            results[name] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'test_mae': test_mae,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }

            print(f"      R² Score: {test_r2:.3f}, MAE: {test_mae:.3f}")

            # 选择最佳模型
            if test_r2 > best_score:
                best_score = test_r2
                best_model = model

        # 保存最佳模型
        self.model = best_model
        self.is_trained = True
        self.model_performance = results

        print(f"Best model selected with R² = {best_score:.3f}")
        return results

    def predict_women_proportion(self, text: str) -> Dict[str, float]:
        """预测单个职位描述的女性申请率"""
        if not self.is_trained:
            print("Model not trained. Training with available data...")
            self.train_model()

        try:
            # 提取特征
            features = self.extract_features(text).reshape(1, -1)

            # 标准化
            features_scaled = self.scaler.transform(features)

            # 预测
            prediction = self.model.predict(features_scaled)[0]

            # 限制在合理范围内
            prediction = max(0.0, min(1.0, prediction))

            return {
                'women_application_rate': prediction,
                'percentage': prediction * 100,
                'confidence': 'medium'  # 简化的置信度
            }

        except Exception as e:
            print(f"Prediction error: {e}")
            return {
                'women_application_rate': 0.5,  # 默认值
                'percentage': 50.0,
                'confidence': 'low'
            }

    def batch_predict(self, texts: List[str]) -> List[Dict]:
        """批量预测多个职位描述的女性申请率"""
        results = []
        for i, text in enumerate(texts):
            prediction = self.predict_women_proportion(text)
            prediction['index'] = i
            results.append(prediction)
        return results

    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性（如果模型支持）"""
        if not self.is_trained or self.model is None:
            return {}

        if hasattr(self.model, 'feature_importances_'):
            importance_dict = {}
            for i, importance in enumerate(self.model.feature_importances_):
                feature_name = self.feature_names[i] if i < len(self.feature_names) else f'feature_{i}'
                importance_dict[feature_name] = importance

            # 按重要性排序
            sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            return sorted_importance
        else:
            return {"message": "Feature importance not available for this model type"}

    def analyze_prediction_factors(self, text: str) -> Dict:
        """分析影响预测结果的因素"""
        prediction_result = self.predict_women_proportion(text)
        bias_analysis = self.bias_detector.analyze_bias_patterns(text)
        feature_importance = self.get_feature_importance()

        # 分析主要影响因素
        factors = []

        if len(bias_analysis.masculine_words) > 3:
            factors.append({
                'factor': 'High masculine word count',
                'impact': 'negative',
                'description': f'Found {len(bias_analysis.masculine_words)} masculine-coded words',
                'words': bias_analysis.masculine_words[:5]
            })

        if len(bias_analysis.inclusive_words) > 2:
            factors.append({
                'factor': 'Inclusive language present',
                'impact': 'positive',
                'description': f'Found {len(bias_analysis.inclusive_words)} inclusive words',
                'words': bias_analysis.inclusive_words[:5]
            })

        if len(bias_analysis.exclusive_words) > 2:
            factors.append({
                'factor': 'Exclusive language detected',
                'impact': 'negative',
                'description': f'Found {len(bias_analysis.exclusive_words)} exclusive expressions',
                'words': bias_analysis.exclusive_words[:3]
            })

        return {
            'prediction': prediction_result,
            'influencing_factors': factors,
            'feature_importance': feature_importance,
            'bias_summary': {
                'overall_bias': bias_analysis.overall_bias,
                'bias_strength': bias_analysis.bias_strength
            }
        }

    def generate_improvement_suggestions(self, text: str) -> Dict:
        """基于预测结果生成改进建议"""
        current_prediction = self.predict_women_proportion(text)
        bias_analysis = self.bias_detector.analyze_bias_patterns(text)

        suggestions = []
        potential_impact = 0

        # 如果女性申请率低于40%，给出具体建议
        if current_prediction['women_application_rate'] < 0.4:

            if len(bias_analysis.masculine_words) > 0:
                suggestions.append({
                    'suggestion': f"Replace masculine-coded words: {', '.join(bias_analysis.masculine_words[:3])}",
                    'expected_impact': '+5-10% women applicants',
                    'priority': 'high'
                })
                potential_impact += 0.07

            if len(bias_analysis.inclusive_words) < 2:
                suggestions.append({
                    'suggestion': "Add inclusive language: 'collaborative', 'supportive', 'diverse team'",
                    'expected_impact': '+3-8% women applicants',
                    'priority': 'medium'
                })
                potential_impact += 0.05

            if len(bias_analysis.exclusive_words) > 1:
                suggestions.append({
                    'suggestion': f"Soften strict requirements: change '{bias_analysis.exclusive_words[0]}' to 'preferred'",
                    'expected_impact': '+2-5% women applicants',
                    'priority': 'medium'
                })
                potential_impact += 0.03

        # 估算改进后的申请率
        improved_rate = min(0.8, current_prediction['women_application_rate'] + potential_impact)

        return {
            'current_rate': current_prediction,
            'suggestions': suggestions,
            'estimated_improved_rate': {
                'women_application_rate': improved_rate,
                'percentage': improved_rate * 100,
                'improvement': (improved_rate - current_prediction['women_application_rate']) * 100
            }
        }

    def save_model(self, filepath: str = 'assets/trained_model.pkl'):
        """保存训练好的模型"""
        if not self.is_trained:
            print("No trained model to save")
            return False

        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'performance': self.model_performance
            }

            # 确保目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            joblib.dump(model_data, filepath)
            print(f"Model saved to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

    def load_model(self, filepath: str = 'assets/trained_model.pkl'):
        """加载预训练模型"""
        try:
            if not os.path.exists(filepath):
                print(f"Model file not found: {filepath}")
                return False

            model_data = joblib.load(filepath)

            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.model_performance = model_data.get('performance', {})
            self.is_trained = True

            print(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def get_model_info(self) -> Dict:
        """获取模型信息和性能指标"""
        if not self.is_trained:
            return {"status": "Model not trained"}

        return {
            "status": "Model trained",
            "model_type": type(self.model).__name__,
            "feature_count": len(self.feature_names),
            "features": self.feature_names,
            "performance": self.model_performance,
            "feature_importance": self.get_feature_importance()
        }


# 创建全局预测器实例
women_predictor = WomenApplicationPredictor()


def get_women_predictor() -> WomenApplicationPredictor:
    """获取女性申请率预测器实例"""
    return women_predictor