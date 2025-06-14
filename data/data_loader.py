# @Versin:1.1
# @Author:Yummy, Esme
"""
数据加载器
支持本地和 HuggingFace 数据集加载，负责加载和预处理三个数据集
"""

import pandas as pd
import os
import sys
from typing import Optional, Tuple
from datasets import load_dataset

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import load_dataset as local_load_dataset, clean_text


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
        self.huggingface_df = None

    def load_all_datasets(self) -> bool:
        """加载所有数据集"""
        try:
            print("📊 开始加载数据集...")

            # 加载合成数据集
            self.synthetic_df = local_load_dataset(os.path.join(self.datasets_path, 'synthetic_vacancies_final.csv'))
            # 加载标注数据集
            self.labeled_df = local_load_dataset(os.path.join(self.datasets_path, 'labeled_vacancies_final.csv'))
            # 加载未标注数据集
            self.unlabeled_df = local_load_dataset(os.path.join(self.datasets_path, 'unlabeled_vacancies_final.csv'))

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

    def load_huggingface_dataset(self, dataset_name: str = "facebook/md_gender_bias", split: str = "train") -> Optional[pd.DataFrame]:
            """从 HuggingFace 加载外部数据集"""
            try:
                print(f"🔗 加载 HuggingFace 数据集: {dataset_name} [{split}]...")
                dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
                df = dataset.to_pandas()

                text_column = "text" if "text" in df.columns else df.columns[0]
                df = df[[text_column]]
                df.rename(columns={text_column: "description"}, inplace=True)

                if "label" in dataset.column_names:
                    df["label"] = dataset["label"]
                if "gender_bias" in df.columns:
                    df.rename(columns={"gender_bias": "women_proportion"}, inplace=True)

                df["description"] = df["description"].apply(clean_text)
                print(f"✅ HuggingFace 加载完成，共 {len(df)} 条记录")
                self.huggingface_df = df
                return df

            except Exception as e:
                print(f"❌ HuggingFace 数据加载失败: {e}")
                return None

    def _print_dataset_info(self):
        """打印数据集信息"""
        print("\n📈 数据集概览:")

        for name, df in [("合成数据集", self.synthetic_df), ("标注数据集", self.labeled_df), ("未标注数据集", self.unlabeled_df)]:
            if df is not None:
                print(f"  • {name}: {len(df)} 行, 列名: {list(df.columns)}")
                if 'women_proportion' in df.columns:
                    print(f"    女性申请比例 - 平均: {df['women_proportion'].mean():.3f}, 范围: {df['women_proportion'].min():.3f}-{df['women_proportion'].max():.3f}")


    def get_combined_training_data(self) -> Optional[pd.DataFrame]:
        """获取合并的训练数据（合成+标注）"""
        try:
            if self.synthetic_df is None or self.labeled_df is None:
                print("❌ 训练数据集未加载")
                return None

            # 标准化列名
            synthetic_df = self.synthetic_df.copy()
            labeled_df = self.labeled_df.copy()
            
            # 确保列名一致
            if 'job_description' in synthetic_df.columns:
                synthetic_df['description'] = synthetic_df['job_description']
            if 'description' not in labeled_df.columns and 'job_description' in labeled_df.columns:
                labeled_df['description'] = labeled_df['job_description']

            # # 选择相同的列
            # common_columns = ['description', 'women_proportion']

            synthetic_subset = synthetic_df[['description', 'women_proportion']]
            labeled_subset = labeled_df[['description', 'women_proportion']]

            # 合并数据
            combined = pd.concat([synthetic_subset, labeled_subset], ignore_index=True)
            # 清理文本数据
            combined['description'] = combined['description'].apply(clean_text)
            # 移除空值
            combined = combined.dropna()
            print(f"✅ 本地训练数据合并完成: {len(combined)} 条记录")
            return combined

        except Exception as e:
            print(f"❌ 本地训练数据合并失败: {e}")
            return None

    def get_combined_training_data_with_hf(self) -> Optional[pd.DataFrame]:
            """获取合并的训练数据（本地 + HuggingFace）"""
            local_df = self.get_combined_training_data()
            hf_df = self.huggingface_df or self.load_huggingface_dataset()

            if local_df is not None and hf_df is not None:
                combined = pd.concat([local_df, hf_df], ignore_index=True)
                combined = combined.dropna()
                print(f"✅ 本地+HuggingFace 训练数据合并完成: {len(combined)} 条记录")
                return combined

            return local_df or hf_df
    
    def get_test_samples(self, n_samples: int = 5) -> list:
        """获取测试样本"""
        samples = []
        for source, df in [("synthetic", self.synthetic_df), ("labeled", self.labeled_df), ("unlabeled", self.unlabeled_df)]:
            if df is not None and len(df) > 0:
                sampled = df.sample(min(n_samples, len(df)))
                for _, row in sampled.iterrows():
                    text = row.get('job_description', row.get('description', ''))
                    samples.append({
                        'text': clean_text(text),
                        'source': source,
                        'women_proportion': row.get('women_proportion', None)
                    })
        return samples[:n_samples]

    def get_dataset_stats(self) -> dict:
        """获取数据集统计信息"""
        stats = {}
        for name, df in [("synthetic", self.synthetic_df), ("labeled", self.labeled_df), ("unlabeled", self.unlabeled_df)]:
            if df is not None:
                stats[name] = {
                    'count': len(df),
                    'columns': list(df.columns),
                    'women_proportion_mean': df['women_proportion'].mean() if 'women_proportion' in df.columns else None
                }
        if self.huggingface_df is not None:
            stats['huggingface'] = {
                'count': len(self.huggingface_df),
                'columns': list(self.huggingface_df.columns)
            }
        return stats


# 创建全局数据加载器实例
data_loader = DataLoader()


def get_data_loader() -> DataLoader:
    """获取数据加载器实例"""
    return data_loader