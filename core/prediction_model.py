"""
Women Application Rate Prediction Model
Optimized version with pre-trained model loading and English output
@P4G
"""

import streamlit as st
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

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import get_text_statistics, calculate_sentiment
from core.bias_detector import get_bias_detector
from data.data_loader import get_data_loader


@st.cache_resource
def load_pretrained_predictor_models():
    """
    Load pre-trained models from disk if available.
    Uses Streamlit caching to ensure models are loaded only once.
    """
    model_dir = "saved_models"
    model_path = os.path.join(model_dir, "women_predictor_model.pkl")

    if os.path.exists(model_path):
        try:
            with st.spinner("Loading pre-trained women predictor model..."):
                model_data = joblib.load(model_path)
                st.success("Pre-trained women predictor model loaded successfully")
                return model_data
        except Exception as e:
            st.warning(f"Failed to load pre-trained model: {str(e)}")
            return None
    else:
        st.info("Pre-trained model not found. Will train online if needed.")
        return None


class WomenApplicationPredictor:
    """Main class for predicting women application rates"""

    def __init__(self):
        """Initialize predictor with features and models"""
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

        # Try to load pre-trained model first
        self._try_load_pretrained_model()

        print("Women Application Predictor initialized")

    def _try_load_pretrained_model(self):
        """Attempt to load pre-trained model if available"""
        model_data = load_pretrained_predictor_models()

        if model_data:
            try:
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.feature_names = model_data['feature_names']
                self.model_performance = model_data.get('performance', {})
                self.is_trained = True
                print("Pre-trained model loaded successfully")
            except Exception as e:
                print(f"Error loading pre-trained model: {e}")
                self.is_trained = False

    def extract_features(self, text: str) -> np.ndarray:
        """Extract machine learning feature vector from text"""
        # Get bias analysis results
        bias_analysis = self.bias_detector.analyze_bias_patterns(text)

        # Get text statistics
        text_stats = get_text_statistics(text)

        # Get sentiment score
        sentiment = calculate_sentiment(text)

        # Build feature vector
        features = [
            len(bias_analysis.masculine_words),  # Masculine word count
            len(bias_analysis.feminine_words),  # Feminine word count
            len(bias_analysis.inclusive_words),  # Inclusive word count
            len(bias_analysis.exclusive_words),  # Exclusive word count
            text_stats['word_count'],  # Text length
            text_stats['avg_sentence_length'],  # Average sentence length
            sentiment,  # Sentiment score
            # Density features
            len(bias_analysis.masculine_words) / max(text_stats['word_count'], 1) * 100,  # Masculine word density
            len(bias_analysis.inclusive_words) / max(text_stats['word_count'], 1) * 100,  # Inclusive word density
        ]

        return np.array(features)

    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data, extract features and labels"""
        print("Preparing training data...")

        # Load datasets
        if not self.data_loader.load_all_datasets():
            raise Exception("Failed to load datasets")

        # Get combined training data
        training_df = self.data_loader.get_combined_training_data()
        if training_df is None or len(training_df) == 0:
            raise Exception("Training data is empty")

        print(f"Training data loaded: {len(training_df)} samples")

        # Extract features and labels
        X = []
        y = []

        # Use progress tracking for Streamlit
        progress_bar = st.progress(0)
        status_text = st.empty()

        total_samples = len(training_df)

        for idx, row in training_df.iterrows():
            try:
                text = row['description']
                women_prop = row['women_proportion']

                # Skip invalid data
                if pd.isna(women_prop) or not isinstance(text, str) or len(text.strip()) < 20:
                    continue

                # Extract features
                features = self.extract_features(text)
                X.append(features)
                y.append(women_prop)

                # Update progress
                if len(X) % 50 == 0:
                    progress = len(X) / total_samples
                    progress_bar.progress(min(progress, 1.0))
                    status_text.text(f"Processed {len(X)} samples...")

            except Exception as e:
                print(f"Error processing sample: {e}")
                continue

        # Clean up progress indicators
        progress_bar.empty()
        status_text.empty()

        X = np.array(X)
        y = np.array(y)

        print(f"Feature extraction completed: {X.shape[0]} valid samples, {X.shape[1]} features")
        print(f"Women application rate distribution: mean={y.mean():.3f}, range=[{y.min():.3f}, {y.max():.3f}]")

        return X, y

    def train_model(self, test_size: float = 0.2) -> Dict:
        """Train prediction model and evaluate performance"""
        print("Starting model training...")

        # Prepare data
        X, y = self.prepare_training_data()

        # Data standardization
        X_scaled = self.scaler.fit_transform(X)

        # Split training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )

        # Train multiple models and select the best
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

        # Show training progress
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, (name, model) in enumerate(models.items()):
            status_text.text(f"Training {name}...")
            progress_bar.progress((i + 1) / len(models))

            # Train model
            model.fit(X_train, y_train)

            # Predictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            # Evaluation
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            test_mae = mean_absolute_error(y_test, test_pred)

            # Cross validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

            results[name] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'test_mae': test_mae,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }

            print(f"      R² Score: {test_r2:.3f}, MAE: {test_mae:.3f}")

            # Select best model
            if test_r2 > best_score:
                best_score = test_r2
                best_model = model

        # Clean up progress indicators
        progress_bar.empty()
        status_text.empty()

        # Save best model
        self.model = best_model
        self.is_trained = True
        self.model_performance = results

        print(f"Best model selected with R² = {best_score:.3f}")

        # Auto-save the trained model
        self.save_model()

        return results

    def predict_women_proportion(self, text: str) -> Dict[str, float]:
        """Predict women application rate for a single job description"""
        if not self.is_trained:
            st.info("Model not trained. Training with available data...")
            try:
                self.train_model()
            except Exception as e:
                st.error(f"Training failed: {str(e)}")
                return {
                    'women_application_rate': 0.5,
                    'percentage': 50.0,
                    'confidence': 'low',
                    'error': 'Training failed'
                }

        try:
            # Extract features
            features = self.extract_features(text).reshape(1, -1)

            # Standardize
            features_scaled = self.scaler.transform(features)

            # Predict
            prediction = self.model.predict(features_scaled)[0]

            # Limit to reasonable range
            prediction = max(0.0, min(1.0, prediction))

            # Calculate confidence based on feature quality
            confidence = self._calculate_confidence(features.flatten())

            return {
                'women_application_rate': prediction,
                'percentage': prediction * 100,
                'confidence': confidence
            }

        except Exception as e:
            print(f"Prediction error: {e}")
            return {
                'women_application_rate': 0.5,  # Default value
                'percentage': 50.0,
                'confidence': 'low',
                'error': str(e)
            }

    def _calculate_confidence(self, features: np.ndarray) -> str:
        """Calculate prediction confidence based on features"""
        try:
            text_length = features[4]  # word_count
            total_gendered_words = features[0] + features[1]  # masculine + feminine

            if text_length > 50 and total_gendered_words > 2:
                return 'high'
            elif text_length > 25 and total_gendered_words > 0:
                return 'medium'
            else:
                return 'low'
        except Exception:
            return 'unknown'

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance if model supports it"""
        if not self.is_trained or self.model is None:
            return {}

        if hasattr(self.model, 'feature_importances_'):
            importance_dict = {}
            for i, importance in enumerate(self.model.feature_importances_):
                feature_name = self.feature_names[i] if i < len(self.feature_names) else f'feature_{i}'
                importance_dict[feature_name] = importance

            # Sort by importance
            sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            return sorted_importance
        else:
            return {"message": "Feature importance not available for this model type"}

    def analyze_prediction_factors(self, text: str) -> Dict:
        """Analyze factors influencing prediction results"""
        prediction_result = self.predict_women_proportion(text)
        bias_analysis = self.bias_detector.analyze_bias_patterns(text)
        feature_importance = self.get_feature_importance()

        # Analyze main influencing factors
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
        """Generate improvement suggestions based on prediction results"""
        current_prediction = self.predict_women_proportion(text)
        bias_analysis = self.bias_detector.analyze_bias_patterns(text)

        suggestions = []
        potential_impact = 0

        # If women application rate is below 40%, give specific suggestions
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

        # Estimate improved application rate
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

    def save_model(self, filepath: str = None):
        """Save trained model to disk"""
        if not self.is_trained:
            print("No trained model to save")
            return False

        if filepath is None:
            model_dir = "saved_models"
            os.makedirs(model_dir, exist_ok=True)
            filepath = os.path.join(model_dir, "women_predictor_model.pkl")

        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'performance': self.model_performance
            }

            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            joblib.dump(model_data, filepath)
            print(f"Model saved to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

    def load_model(self, filepath: str = None):
        """Load pre-trained model from disk"""
        if filepath is None:
            filepath = os.path.join("saved_models", "women_predictor_model.pkl")

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
        """Get model information and performance metrics"""
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


# Create cached global predictor instance
@st.cache_resource
def get_cached_women_predictor():
    """Get cached women predictor instance"""
    return WomenApplicationPredictor()


def get_women_predictor() -> WomenApplicationPredictor:
    """Get women application rate predictor instance"""
    return get_cached_women_predictor()