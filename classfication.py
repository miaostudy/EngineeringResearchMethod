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
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

DATA_PATH = "defect_prediction_dataset.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42
FEATURE_SELECT_K = 8
FIGURE_DIR = "classification_figures"

import os

os.makedirs(FIGURE_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_and_preprocess_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(['filename', 'is_buggy', 'bug_count'], axis=1)
    y = df['is_buggy']
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    return X_train, X_test, y_train, y_test, feature_names


def build_models():
    feature_engineering = [
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(f_classif, k=FEATURE_SELECT_K))
    ]

    classifiers = {
        "逻辑回归": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        "决策树": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "随机森林": RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100),
        "支持向量机": SVC(random_state=RANDOM_STATE, probability=True, kernel='rbf'),
        "XGB分类": xgb.XGBClassifier(random_state=RANDOM_STATE, n_estimators=100, use_label_encoder=False,
                                     eval_metric='logloss'),
        "LGB分类": lgb.LGBMClassifier(random_state=RANDOM_STATE, n_estimators=100)
    }

    pipelines = {}
    for name, clf in classifiers.items():
        pipelines[name] = ImbPipeline([
            *feature_engineering,
            ('smote', SMOTE(random_state=RANDOM_STATE)),  # 已有上采样，保留
            ('classifier', clf)
        ])

    return pipelines


def train_and_evaluate(pipelines, X_train, X_test, y_train, y_test, feature_names):
    results = {}
    for name, pipeline in pipelines.items():
        print(f"\n{'=' * 50}")
        print(f"模型: {name}")
        print(f"{'=' * 50}")
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_pred_prob = pipeline.predict_proba(X_test)[:, 1]

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_prob)

        results[name] = {
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(f1, 4),
            "AUC": round(auc, 4),
            "y_pred": y_pred,
            "y_pred_prob": y_pred_prob,
            "pipeline": pipeline
        }

        print("分类报告:")
        print(classification_report(y_test, y_pred, target_names=['非Bug文件', 'Bug文件']))
        print(f"AUC: {auc:.4f}")

    results_df = pd.DataFrame(results).T
    results_df = results_df[['Precision', 'Recall', 'F1-Score', 'AUC']]

    print(f"\n{'=' * 50}")
    print("所有模型性能对比:")
    print(results_df.round(4))

    results_df.to_csv("./classification_results.csv", encoding='utf-8-sig')

    return results, results_df, feature_names


def plot_feature_importance(results, feature_names, X_train, y_train):
    best_model_name = max(results.keys(), key=lambda x: results[x]['F1-Score'])
    best_pipeline = results[best_model_name]['pipeline']

    selector = best_pipeline.named_steps['selector']
    selected_feature_indices = selector.get_support(indices=True)
    selected_feature_names = [feature_names[i] for i in selected_feature_indices]

    classifier = best_pipeline.named_steps['classifier']

    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
    else:
        X_processed = best_pipeline.named_steps['smote'].fit_resample(
            best_pipeline.named_steps['selector'].transform(
                best_pipeline.named_steps['scaler'].transform(X_train)
            ), y_train
        )[0]
        result = permutation_importance(
            classifier, X_processed, y_train, n_repeats=10,
            random_state=RANDOM_STATE, n_jobs=-1
        )
        importances = result.importances_mean

    fig, ax = plt.subplots(figsize=(12, 8))
    sorted_idx = np.argsort(importances)

    ax.barh(range(len(sorted_idx)), importances[sorted_idx], align='center', color='#ff7f0e', alpha=0.8)
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([selected_feature_names[i] for i in sorted_idx])
    ax.set_xlabel('特征重要性', fontsize=12)
    ax.set_title(f'{best_model_name} - 特征重要性排名', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    for i, v in enumerate(importances[sorted_idx]):
        ax.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'feature_importance.png'), dpi=300, bbox_inches='tight')


def visualize_results(results, results_df, X_test, y_test, feature_names, X_train, y_train):
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

    best_clf_name = results_df['F1-Score'].idxmax()
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

    fig, ax = plt.subplots(figsize=(10, 8))
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['y_pred_prob'])
        ax.plot(fpr, tpr, label=f'{name} (AUC = {result["AUC"]:.4f})', linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', label='随机猜测 (AUC = 0.5)', linewidth=1)
    ax.set_xlabel('假正例率 (FPR)', fontsize=12)
    ax.set_ylabel('真正例率 (TPR)', fontsize=12)
    ax.set_title('不同分类器的ROC曲线', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'roc_curves.png'), dpi=300, bbox_inches='tight')

    plot_feature_importance(results, feature_names, X_train, y_train)


def main():
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data()
    pipelines = build_models()
    results, results_df, feature_names = train_and_evaluate(pipelines, X_train, X_test, y_train, y_test, feature_names)
    visualize_results(results, results_df, X_test, y_test, feature_names, X_train, y_train)  # 传递X_train和y_train


if __name__ == "__main__":
    main()
