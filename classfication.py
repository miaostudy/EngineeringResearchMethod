import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings

warnings.filterwarnings('ignore')

DATA_PATH = "./defect_prediction_dataset.csv"
TEST_SIZE = 0.3
RANDOM_STATE = 42
FEATURE_SELECT_K = 8
FIGURE_DIR = "./classification_figures"

import os

os.makedirs(FIGURE_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_and_preprocess_data():
    df = pd.read_csv(DATA_PATH)
    print(f"数据集大小: {df.shape}")
    print(f"\n类别分布（原始）:")
    print(df['is_buggy'].value_counts())
    print(f"类别不平衡比例: {round(df['is_buggy'].value_counts()[0] / df['is_buggy'].value_counts()[1], 1)}:1")

    # 分离特征和标签
    X = df.drop(['filename', 'is_buggy', 'bug_count'], axis=1)
    y = df['is_buggy']
    feature_names = X.columns.tolist()
    print(f"\n特征列表: {feature_names}")

    # 划分训练集和测试集（分层抽样，保持类别分布）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")
    print(f"训练集类别分布: {y_train.value_counts().to_dict()}")
    print(f"测试集类别分布: {y_test.value_counts().to_dict()}")

    return X_train, X_test, y_train, y_test, feature_names


# ================= 构建模型 pipeline =================
def build_models():
    """构建多个分类器的pipeline（包含特征工程+模型）"""
    # 特征工程：标准化 + 特征选择
    feature_engineering = [
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(f_classif, k=FEATURE_SELECT_K))
    ]

    # 定义多个分类器
    classifiers = {
        "逻辑回归": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        "决策树": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "随机森林": RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100),
        "支持向量机": SVC(random_state=RANDOM_STATE, probability=True, kernel='rbf')
    }

    # 构建pipeline（包含SMOTE上采样，仅在训练集生效）
    pipelines = {}
    for name, clf in classifiers.items():
        pipelines[name] = ImbPipeline([
            *feature_engineering,
            ('smote', SMOTE(random_state=RANDOM_STATE)),
            ('classifier', clf)
        ])

    return pipelines


# ================= 训练与评估 =================
def train_and_evaluate(pipelines, X_train, X_test, y_train, y_test, feature_names):
    """训练模型并评估性能"""
    results = {}

    print("\n" + "=" * 80)
    print("开始训练和评估分类器...")
    print("=" * 80)

    for name, pipeline in pipelines.items():
        print(f"\n【{name}】")
        print("-" * 50)

        # 训练模型
        pipeline.fit(X_train, y_train)

        # 预测
        y_pred = pipeline.predict(X_test)
        y_pred_prob = pipeline.predict_proba(X_test)[:, 1]  # 正类概率（用于AUC）

        # 计算指标
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_prob)

        # 保存结果
        results[name] = {
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(f1, 4),
            "AUC": round(auc, 4),
            "y_pred": y_pred,
            "y_pred_prob": y_pred_prob,
            "pipeline": pipeline
        }

        # 打印详细报告
        print("分类报告:")
        print(classification_report(y_test, y_pred, target_names=['非Bug文件', 'Bug文件']))
        print(f"AUC: {auc:.4f}")

    # 保存结果到DataFrame
    results_df = pd.DataFrame(results).T
    results_df = results_df[['Precision', 'Recall', 'F1-Score', 'AUC']]
    print("\n" + "=" * 80)
    print("所有分类器性能对比:")
    print("=" * 80)
    print(results_df.round(4))

    # 保存结果到CSV
    results_df.to_csv("./classification_results.csv", encoding='utf-8-sig')
    print(f"\n性能结果已保存到: ./classification_results.csv")

    return results, results_df, feature_names


# ================= 可视化 =================
def visualize_results(results, results_df, X_test, y_test, feature_names):
    """可视化结果（混淆矩阵、ROC曲线、指标对比）"""
    print("\n开始生成可视化图表...")

    # 1. 指标对比柱状图
    fig, ax = plt.subplots(figsize=(12, 8))
    metrics = ['Precision', 'Recall', 'F1-Score', 'AUC']
    x = np.arange(len(results_df.index))
    width = 0.2
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, results_df[metric], width, label=metric, color=colors[i], alpha=0.8)

    ax.set_xlabel('分类器', fontsize=12)
    ax.set_ylabel('指标值', fontsize=12)
    ax.set_title('不同分类器性能指标对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(results_df.index, rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"✅ 指标对比图已保存")

    best_clf_name = results_df['F1-Score'].idxmax()
    print(f"\n选择性能最优分类器进行详细可视化: {best_clf_name}")
    best_results = results[best_clf_name]
    cm = confusion_matrix(y_test, best_results['y_pred'])

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('预测标签', fontsize=12)
    ax.set_ylabel('真实标签', fontsize=12)
    ax.set_title(f'{best_clf_name} - 混淆矩阵', fontsize=14, fontweight='bold')
    ax.set_xticklabels(['非Bug文件', 'Bug文件'])
    ax.set_yticklabels(['非Bug文件', 'Bug文件'])
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    print(f"✅ 混淆矩阵已保存")

    # 3. ROC曲线（所有分类器）
    fig, ax = plt.subplots(figsize=(10, 8))
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['y_pred_prob'])
        ax.plot(fpr, tpr, label=f'{name} (AUC = {result["AUC"]:.4f})', linewidth=2)

    # 随机猜测线
    ax.plot([0, 1], [0, 1], 'k--', label='随机猜测 (AUC = 0.5)', linewidth=1)
    ax.set_xlabel('假正例率 (FPR)', fontsize=12)
    ax.set_ylabel('真正例率 (TPR)', fontsize=12)
    ax.set_title('不同分类器的ROC曲线', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    print(f"✅ ROC曲线已保存")

    # 4. 特征重要性（随机森林）
    if '随机森林' in results:
        rf_pipeline = results['随机森林']['pipeline']
        selector = rf_pipeline.named_steps['selector']
        classifier = rf_pipeline.named_steps['classifier']

        # 获取选中的特征名称
        selected_idx = selector.get_support(indices=True)
        selected_features = [feature_names[i] for i in selected_idx]
        feature_importance = classifier.feature_importances_

        # 可视化特征重要性
        fig, ax = plt.subplots(figsize=(10, 6))
        sorted_idx = np.argsort(feature_importance)
        ax.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([selected_features[i] for i in sorted_idx])
        ax.set_xlabel('特征重要性', fontsize=12)
        ax.set_title('随机森林 - 特征重要性排序', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_DIR, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        print(f"✅ 特征重要性图已保存")

    print(f"\n所有可视化图表已保存到: {FIGURE_DIR}")


# ================= 主流程 =================
def main():
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data()

    pipelines = build_models()

    results, results_df, feature_names = train_and_evaluate(pipelines, X_train, X_test, y_train, y_test, feature_names)

    visualize_results(results, results_df, X_test, y_test, feature_names)

if __name__ == "__main__":
    main()
