# @Versin:1.0
# @Author:Yummy
"""
数据加载器
负责加载和预处理三个数据集
"""

import pandas as pd
import os
import sys
from typing import Optional, Tuple

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import load_dataset, clean_text


class DataLoader:
    """数据加载和管理类"""

    def __init__(self, datasets_path: str = "datasets"):
        """
        初始化数据加载器

        Args:
            datasets_path: 数据集文件夹路径
        """
        self.datasets_path = datasets_path
        self.synthetic_df = None
        self.labeled_df = None
        self.unlabeled_df = None

    def load_all_datasets(self) -> bool:
        """加载所有数据集"""
        try:
            print("📊 开始加载数据集...")

            # 加载合成数据集
            synthetic_path = os.path.join(self.datasets_path, 'synthetic_vacancies_final.csv')
            self.synthetic_df = load_dataset(synthetic_path)

            # 加载标注数据集
            labeled_path = os.path.join(self.datasets_path, 'labeled_vacancies_final.csv')
            self.labeled_df = load_dataset(labeled_path)

            # 加载未标注数据集
            unlabeled_path = os.path.join(self.datasets_path, 'unlabeled_vacancies_final.csv')
            self.unlabeled_df = load_dataset(unlabeled_path)

            # 验证加载结果
            if all([df is not None for df in [self.synthetic_df, self.labeled_df, self.unlabeled_df]]):
                print("✅ 所有数据集加载成功")
                self._print_dataset_info()
                return True
            else:
                print("❌ 部分数据集加载失败")
                return False

        except Exception as e:
            print(f"❌ 数据加载异常: {e}")
            return False

    def _print_dataset_info(self):
        """打印数据集信息"""
        print("\n📈 数据集概览:")

        if self.synthetic_df is not None:
            print(f"  • 合成数据集: {len(self.synthetic_df)} 行")
            print(f"    列名: {list(self.synthetic_df.columns)}")
            if 'women_proportion' in self.synthetic_df.columns:
                women_prop = self.synthetic_df['women_proportion']
                print(
                    f"    女性申请比例 - 平均: {women_prop.mean():.3f}, 范围: {women_prop.min():.3f}-{women_prop.max():.3f}")

        if self.labeled_df is not None:
            print(f"  • 标注数据集: {len(self.labeled_df)} 行")
            print(f"    列名: {list(self.labeled_df.columns)}")
            if 'women_proportion' in self.labeled_df.columns:
                women_prop = self.labeled_df['women_proportion']
                print(
                    f"    女性申请比例 - 平均: {women_prop.mean():.3f}, 范围: {women_prop.min():.3f}-{women_prop.max():.3f}")

        if self.unlabeled_df is not None:
            print(f"  • 未标注数据集: {len(self.unlabeled_df)} 行")
            print(f"    列名: {list(self.unlabeled_df.columns)}")

    def get_combined_training_data(self) -> Optional[pd.DataFrame]:
        """获取合并的训练数据（合成+标注）"""
        try:
            if self.synthetic_df is None or self.labeled_df is None:
                print("❌ 训练数据集未加载")
                return None

            # 标准化列名
            synthetic_clean = self.synthetic_df.copy()
            labeled_clean = self.labeled_df.copy()

            # 确保列名一致
            if 'job_description' in synthetic_clean.columns:
                synthetic_clean['description'] = synthetic_clean['job_description']

            if 'description' in labeled_clean.columns:
                labeled_clean['description'] = labeled_clean['description']

            # 选择相同的列
            common_columns = ['description', 'women_proportion']

            synthetic_subset = synthetic_clean[common_columns]
            labeled_subset = labeled_clean[common_columns]

            # 合并数据
            combined_df = pd.concat([synthetic_subset, labeled_subset], ignore_index=True)

            # 清理文本数据
            combined_df['description'] = combined_df['description'].apply(clean_text)

            # 移除空值
            combined_df = combined_df.dropna()

            print(f"✅ 合并训练数据: {len(combined_df)} 行")
            return combined_df

        except Exception as e:
            print(f"❌ 合并训练数据失败: {e}")
            return None

    def get_test_samples(self, n_samples: int = 5) -> list:
        """获取测试样本"""
        try:
            samples = []

            # 从每个数据集取样本
            if self.synthetic_df is not None and len(self.synthetic_df) > 0:
                sample = self.synthetic_df.sample(min(2, len(self.synthetic_df)))
                for _, row in sample.iterrows():
                    text = row.get('job_description', row.get('description', ''))
                    samples.append({
                        'text': clean_text(text),
                        'source': 'synthetic',
                        'women_proportion': row.get('women_proportion', None)
                    })

            if self.labeled_df is not None and len(self.labeled_df) > 0:
                sample = self.labeled_df.sample(min(2, len(self.labeled_df)))
                for _, row in sample.iterrows():
                    text = row.get('description', row.get('job_description', ''))
                    samples.append({
                        'text': clean_text(text),
                        'source': 'labeled',
                        'women_proportion': row.get('women_proportion', None)
                    })

            if self.unlabeled_df is not None and len(self.unlabeled_df) > 0:
                sample = self.unlabeled_df.sample(min(1, len(self.unlabeled_df)))
                for _, row in sample.iterrows():
                    text = row.get('job_description', row.get('description', ''))
                    samples.append({
                        'text': clean_text(text),
                        'source': 'unlabeled',
                        'women_proportion': None
                    })

            return samples[:n_samples]

        except Exception as e:
            print(f"❌ 获取测试样本失败: {e}")
            return []

    def get_dataset_stats(self) -> dict:
        """获取数据集统计信息"""
        stats = {}

        if self.synthetic_df is not None:
            stats['synthetic'] = {
                'count': len(self.synthetic_df),
                'columns': list(self.synthetic_df.columns),
                'women_proportion_mean': self.synthetic_df[
                    'women_proportion'].mean() if 'women_proportion' in self.synthetic_df.columns else None
            }

        if self.labeled_df is not None:
            stats['labeled'] = {
                'count': len(self.labeled_df),
                'columns': list(self.labeled_df.columns),
                'women_proportion_mean': self.labeled_df[
                    'women_proportion'].mean() if 'women_proportion' in self.labeled_df.columns else None
            }

        if self.unlabeled_df is not None:
            stats['unlabeled'] = {
                'count': len(self.unlabeled_df),
                'columns': list(self.unlabeled_df.columns)
            }

        return stats


# 创建全局数据加载器实例
data_loader = DataLoader()


def get_data_loader() -> DataLoader:
    """获取数据加载器实例"""
    return data_loader