
"""
Quick Demo Script
Run this to test core functionality without UI
"""

import sys
import os
sys.path.append('.')

def quick_demo():
    print("🎯 Quick GenderLens AI Demo")
    print("=" * 40)

    # Test text
    biased_text = """
    We need an aggressive ninja developer who must dominate the competition. 
    This demanding role requires a rockstar who can work independently.
    """

    inclusive_text = """
    We welcome a collaborative developer to join our diverse team. 
    We offer flexible work arrangements and support professional development.
    """

    try:
        # Import modules
        from core.bias_detector import get_bias_detector
        from core.inclusivity_scorer import get_inclusivity_scorer

        detector = get_bias_detector()
        scorer = get_inclusivity_scorer()

        print("\n🔍 Testing Biased Text:")
        bias_result = detector.analyze_bias_patterns(biased_text)
        score_result = scorer.score_job_description(biased_text)

        print(f"   Bias: {bias_result.overall_bias}")
        print(f"   Score: {score_result.overall_score:.1f}")
        print(f"   Masculine words: {bias_result.masculine_words}")

        print("\n✅ Testing Inclusive Text:")
        bias_result2 = detector.analyze_bias_patterns(inclusive_text)
        score_result2 = scorer.score_job_description(inclusive_text)

        print(f"   Bias: {bias_result2.overall_bias}")
        print(f"   Score: {score_result2.overall_score:.1f}")
        print(f"   Inclusive words: {bias_result2.inclusive_words}")

        print("\n🎉 Demo completed successfully!")
        print("✅ Core algorithms working")
        print("🚀 Ready for UI testing!")

    except Exception as e:
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    quick_demo()
