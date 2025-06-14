"""
UI功能测试
验证Streamlit应用的基本功能
运行命令: streamlit run main.py --server.port 8501
"""

import subprocess
import sys
import time
import requests
from pathlib import Path


def test_streamlit_installation():
    """测试Streamlit是否正确安装"""
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
        print(f"   Version: {st.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False


def test_ui_components():
    """测试UI组件模块"""
    try:
        from ui.components import (
            create_score_gauge, create_word_cloud_chart,
            display_metric_card, display_suggestion_card
        )
        print("✅ UI components imported successfully")

        # 测试评分仪表盘创建
        fig = create_score_gauge(75.5, "Test Score")
        print(f"   Score gauge created: {type(fig)}")

        # 测试词汇图表
        test_words = {
            'masculine': ['aggressive', 'competitive'],
            'feminine': ['collaborative'],
            'inclusive': ['diverse', 'welcoming'],
            'exclusive': ['demanding']
        }

        word_fig = create_word_cloud_chart(test_words)
        print(f"   Word chart created: {type(word_fig)}")

        return True

    except Exception as e:
        print(f"❌ UI components test failed: {e}")
        return False


def test_dashboard_module():
    """测试仪表板模块"""
    try:
        from ui.dashboard import create_main_dashboard
        print("✅ Dashboard module imported successfully")
        return True
    except Exception as e:
        print(f"❌ Dashboard module test failed: {e}")
        return False


def test_main_app_structure():
    """测试主应用结构"""
    try:
        # 检查main.py文件是否存在
        main_file = Path("main.py")
        if not main_file.exists():
            print("❌ main.py file not found")
            return False

        print("✅ main.py file exists")

        # 尝试导入主要函数（不运行Streamlit）
        import importlib.util

        spec = importlib.util.spec_from_file_location("main", "main.py")
        main_module = importlib.util.module_from_spec(spec)

        # 检查是否包含主要函数
        with open("main.py", "r", encoding="utf-8") as f:
            content = f.read()

        required_components = [
            "def main():",
            "st.set_page_config",
            "def analyze_job_description",
            "def display_detailed_analysis"
        ]

        missing_components = []
        for component in required_components:
            if component not in content:
                missing_components.append(component)

        if missing_components:
            print(f"❌ Missing components: {missing_components}")
            return False

        print("✅ Main app structure validated")
        return True

    except Exception as e:
        print(f"❌ Main app structure test failed: {e}")
        return False


def test_app_dependencies():
    """测试应用依赖"""
    required_modules = [
        'streamlit',
        'plotly',
        'pandas',
        'numpy'
    ]

    missing_modules = []

    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module} available")
        except ImportError:
            missing_modules.append(module)
            print(f"❌ {module} missing")

    if missing_modules:
        print(f"\n❌ Missing dependencies: {missing_modules}")
        print("Run: pip install streamlit plotly pandas numpy")
        return False

    return True


def test_project_structure():
    """测试项目结构完整性"""
    required_files = [
        "main.py",
        "ui/__init__.py",
        "ui/components.py",
        "ui/dashboard.py",
        "core/bias_detector.py",
        "core/inclusivity_scorer.py",
        "core/prediction_model.py",
        "utils/constants.py",
        "utils/helpers.py",
        "data/data_loader.py",
        "assets/bias_words.json"
    ]

    missing_files = []

    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")

    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        return False

    print("\n✅ All required files present")
    return True


def generate_test_command():
    """生成测试命令"""
    print("\n🚀 To test the UI manually:")
    print("=" * 50)
    print("1. Run the Streamlit app:")
    print("   streamlit run main.py")
    print("\n2. Open your browser to:")
    print("   http://localhost:8501")
    print("\n3. Test these features:")
    print("   • Enter job description text")
    print("   • Click 'Analyze' button")
    print("   • Check all tabs work")
    print("   • Try demo examples")
    print("   • Test sidebar options")


def create_quick_demo():
    """创建快速演示脚本"""
    demo_script = '''
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

        print("\\n🔍 Testing Biased Text:")
        bias_result = detector.analyze_bias_patterns(biased_text)
        score_result = scorer.score_job_description(biased_text)

        print(f"   Bias: {bias_result.overall_bias}")
        print(f"   Score: {score_result.overall_score:.1f}")
        print(f"   Masculine words: {bias_result.masculine_words}")

        print("\\n✅ Testing Inclusive Text:")
        bias_result2 = detector.analyze_bias_patterns(inclusive_text)
        score_result2 = scorer.score_job_description(inclusive_text)

        print(f"   Bias: {bias_result2.overall_bias}")
        print(f"   Score: {score_result2.overall_score:.1f}")
        print(f"   Inclusive words: {bias_result2.inclusive_words}")

        print("\\n🎉 Demo completed successfully!")
        print("✅ Core algorithms working")
        print("🚀 Ready for UI testing!")

    except Exception as e:
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    quick_demo()
'''

    with open("quick_demo.py", "w", encoding="utf-8") as f:
        f.write(demo_script)

    print("✅ Created quick_demo.py")
    print("   Run: python quick_demo.py")


def main():
    """主测试函数"""
    print("🧪 UI Development Testing")
    print("=" * 50)

    tests = [
        ("Project Structure", test_project_structure),
        ("App Dependencies", test_app_dependencies),
        ("Streamlit Installation", test_streamlit_installation),
        ("Main App Structure", test_main_app_structure),
        ("UI Components", test_ui_components),
        ("Dashboard Module", test_dashboard_module)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🔧 Testing {test_name}:")
        try:
            if test_func():
                print(f"✅ {test_name} test passed")
                passed += 1
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test error: {e}")
        print("-" * 30)

    print(f"\n📊 Test Results: {passed}/{total} passed")

    if passed == total:
        print("🎉 All UI tests passed!")
        print("✅ Step 4 completed successfully")

        # 创建演示文件
        create_quick_demo()

        # 生成测试命令
        generate_test_command()

        print("\n🚀 Ready for Step 5: Integration & Testing")
        return True
    else:
        print("❌ Some UI tests failed")
        print("Please fix the issues before proceeding")
        return False


if __name__ == "__main__":
    main()