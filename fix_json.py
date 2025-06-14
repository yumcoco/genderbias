"""
修复bias_words.json文件
运行命令: python fix_json.py
"""

import json
import os


def create_bias_words_json():
    """创建偏向词汇JSON文件"""

    bias_words_data = {
        "masculine_coded": {
            "competitive_terms": [
                "aggressive", "assertive", "competitive", "dominant", "forceful",
                "ambitious", "driven", "determined", "decisive", "confident"
            ],
            "independence_terms": [
                "independent", "individual", "self-reliant", "autonomous", "leader",
                "outspoken", "strong", "fearless", "bold", "challenging"
            ],
            "tech_slang": [
                "ninja", "rockstar", "guru", "wizard", "champion", "warrior",
                "hero", "master", "expert", "hacker", "superstar"
            ]
        },
        "feminine_coded": {
            "collaborative_terms": [
                "collaborative", "cooperative", "supportive", "understanding",
                "interdependent", "team-oriented", "together", "community", "share"
            ],
            "nurturing_terms": [
                "empathetic", "nurturing", "considerate", "caring", "patient",
                "gentle", "kind", "thoughtful", "compassionate", "sensitive"
            ],
            "communication_terms": [
                "communicate", "listen", "responsive", "interpersonal", "relationship",
                "connect", "engage", "interact", "dialogue"
            ]
        },
        "inclusive_terms": {
            "diversity_words": [
                "diverse", "inclusive", "welcoming", "belonging", "equity",
                "fair", "equal opportunity", "accessible", "accommodation"
            ],
            "growth_words": [
                "development", "learning", "growth", "mentorship", "training",
                "career advancement", "professional development", "opportunity"
            ],
            "balance_words": [
                "flexible", "work-life balance", "remote", "hybrid", "flexible hours",
                "family-friendly", "wellness", "support"
            ]
        },
        "exclusive_indicators": {
            "pressure_terms": [
                "demanding", "intense", "fast-paced", "high-pressure", "stressful",
                "aggressive deadlines", "tight deadlines", "demanding environment"
            ],
            "strict_requirements": [
                "must have", "required", "essential", "mandatory", "critical",
                "absolutely necessary", "non-negotiable", "strict requirements"
            ],
            "limiting_phrases": [
                "only consider", "exclusively", "solely", "limited to",
                "perfect candidate", "ideal candidate must", "we only accept"
            ]
        },
        "neutral_alternatives": {
            "aggressive": ["proactive", "goal-oriented", "results-driven"],
            "dominant": ["influential", "impactful", "leading"],
            "ninja": ["expert", "specialist", "professional"],
            "rockstar": ["talented", "skilled", "exceptional"],
            "demanding": ["challenging", "engaging", "dynamic"],
            "must have": ["preferred", "desired", "valuable"],
            "required": ["preferred", "beneficial", "advantageous"]
        }
    }

    # 确保assets目录存在
    assets_dir = "assets"
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)
        print(f"📁 创建目录: {assets_dir}")

    # 写入JSON文件
    json_file = os.path.join(assets_dir, "bias_words.json")
    try:
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(bias_words_data, f, indent=2, ensure_ascii=False)

        print(f"✅ 成功创建: {json_file}")

        # 验证文件
        with open(json_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)

        print(f"✅ JSON文件验证通过")
        print(f"📊 包含类别: {list(loaded_data.keys())}")

        # 统计词汇数量
        total_words = 0
        for category, subcategories in loaded_data.items():
            if isinstance(subcategories, dict):
                for subcat, words in subcategories.items():
                    if isinstance(words, list):
                        total_words += len(words)
                        print(f"   • {category}.{subcat}: {len(words)} 个词")

        print(f"📈 总计: {total_words} 个词汇")
        return True

    except Exception as e:
        print(f"❌ 创建JSON文件失败: {e}")
        return False


def test_json_loading():
    """测试JSON文件加载"""
    try:
        # 测试直接加载
        with open('assets/bias_words.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

        print("✅ 直接加载测试通过")

        # 测试通过helpers模块加载
        import sys
        sys.path.append('.')
        from utils.helpers import load_bias_words

        bias_data = load_bias_words()
        print("✅ helpers模块加载测试通过")
        print(f"📚 加载的类别: {list(bias_data.keys())}")

        return True

    except Exception as e:
        print(f"❌ JSON加载测试失败: {e}")
        return False


if __name__ == "__main__":
    print("🔧 修复bias_words.json文件")
    print("=" * 40)

    # 创建JSON文件
    if create_bias_words_json():
        print("\n🧪 测试文件加载...")
        if test_json_loading():
            print("\n🎉 JSON文件修复完成！")
            print("✅ 现在可以重新运行: python test_config.py")
        else:
            print("\n❌ JSON文件加载测试失败")
    else:
        print("\n❌ JSON文件创建失败")