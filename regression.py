import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance
import warnings

warnings.filterwarnings('ignore')

DATA_PATH = "defect_prediction_dataset.csv"
TEST_SIZE = 0.3
RANDOM_STATE = 42
FEATURE_SELECT_K = 8
FIGURE_DIR = "regression_figures"

import os

os.makedirs(FIGURE_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_and_preprocess_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(['filename', 'is_buggy', 'bug_count'], axis=1)
    y = df['bug_count']
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test, feature_names

def build_models():
    feature_engineering = [
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(f_regression, k=FEATURE_SELECT_K))
    ]

    regressors = {
        "线性回归": LinearRegression(),
        "决策树回归": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "随机森林回归": RandomForestRegressor(random_state=RANDOM_STATE, n_estimators=100),
        "支持向量回归": SVR(kernel='rbf')
    }

    pipelines = {}
    for name, reg in regressors.items():
        pipelines[name] = Pipeline([
            *feature_engineering,
            ('regressor', reg)
        ])

    return pipelines

from sklearn.pipeline import Pipeline

def train_and_evaluate(pipelines, X_train, X_test, y_train, y_test, feature_names):
    results = {}

    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        results[name] = {
            "R2 Score": round(r2, 4),
            "MSE": round(mse, 4),
            "RMSE": round(rmse, 4),
            "y_pred": y_pred,
            "pipeline": pipeline
        }

        print(f"R2 Score: {r2:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"预测值范围: [{y_pred.min():.2f}, {y_pred.max():.2f}]")
        print(f"真实值范围: [{y_test.min():.2f}, {y_test.max():.2f}]")

    results_df = pd.DataFrame(results).T
    results_df = results_df[['R2 Score', 'MSE', 'RMSE']]
    print(results_df.round(4))

    results_df.to_csv("./regression_results.csv", encoding='utf-8-sig')

    return results, results_df, feature_names

def visualize_results(results, results_df, X_test, y_test, feature_names):
    fig, ax = plt.subplots(figsize=(12, 8))
    metrics = ['R2 Score', 'MSE', 'RMSE']
    x = np.arange(len(results_df.index))
    width = 0.25
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, results_df[metric], width, label=metric, color=colors[i], alpha=0.8)

    ax.set_xlabel('回归器', fontsize=12)
    ax.set_ylabel('指标值', fontsize=12)
    ax.set_title('不同回归器性能指标对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(results_df.index, rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')

    best_reg_name = results_df['R2 Score'].idxmax()
    best_results = results[best_reg_name]
    y_pred = best_results['y_pred']

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(y_test, y_pred, alpha=0.6, color='#1f77b4', label='预测值')

    max_val = max(y_test.max(), y_pred.max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='理想预测线 (y=x)')

    ax.text(0.05, 0.95, f'R2 Score = {best_results["R2 Score"]:.4f}',
            transform=ax.transAxes, fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('真实Bug数量', fontsize=12)
    ax.set_ylabel('预测Bug数量', fontsize=12)
    ax.set_title(f'{best_reg_name} - 预测值vs真实值', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'pred_vs_true.png'), dpi=300, bbox_inches='tight')

    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(y_pred, residuals, alpha=0.6, color='#ff7f0e')
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('预测Bug数量', fontsize=12)
    ax.set_ylabel('残差（真实值-预测值）', fontsize=12)
    ax.set_title(f'{best_reg_name} - 残差分布图', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'residual_plot.png'), dpi=300, bbox_inches='tight')


def main():
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data()
    pipelines = build_models()
    results, results_df, feature_names = train_and_evaluate(pipelines, X_train, X_test, y_train, y_test, feature_names)
    visualize_results(results, results_df, X_test, y_test, feature_names)


if __name__ == "__main__":
    main()
