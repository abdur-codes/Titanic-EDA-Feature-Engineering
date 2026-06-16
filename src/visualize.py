import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_missing_data(original: pd.DataFrame, cleaned: pd.DataFrame, output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, df, title in zip(axes, [original, cleaned], ['Before Imputation', 'After Imputation']):
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values()
        if len(missing) == 0:
            ax.text(0.5, 0.5, 'No missing values', ha='center', va='center', fontsize=14)
            ax.set_title(title)
            continue
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(missing)))
        bars = ax.barh(range(len(missing)), missing.values, color=colors)
        ax.set_yticks(range(len(missing)))
        ax.set_yticklabels(missing.index, fontsize=9)
        ax.set_xlabel('Missing Count')
        ax.set_title(title)
        for bar, val in zip(bars, missing.values):
            ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                    str(val), va='center', fontsize=9)

    plt.suptitle('Missing Data Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = Path(output_dir) / '01_missing_data.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_outliers(df_original: pd.DataFrame, df_cleaned: pd.DataFrame, output_dir: str):
    numeric_cols = df_original.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ['survived', 'sibsp', 'parch', 'pclass']
    plot_cols = [c for c in numeric_cols if c not in exclude][:4]

    fig, axes = plt.subplots(2, len(plot_cols), figsize=(14, 6))

    for i, col in enumerate(plot_cols):
        bp1 = axes[0, i].boxplot(df_original[col].dropna(), vert=True, patch_artist=True)
        bp1['boxes'][0].set_facecolor('#ff6b6b')
        axes[0, i].set_title(f'{col} (Before)', fontsize=10)
        axes[0, i].tick_params(axis='x', rotation=45)

        if col in df_cleaned.columns:
            bp2 = axes[1, i].boxplot(df_cleaned[col].dropna(), vert=True, patch_artist=True)
            bp2['boxes'][0].set_facecolor('#51cf66')
            axes[1, i].set_title(f'{col} (After)', fontsize=10)

    plt.suptitle('Outlier Treatment: Box Plot Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = Path(output_dir) / '02_outlier_treatment.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_correlation_matrix(df_before: pd.DataFrame, df_after: pd.DataFrame, output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, df, title in zip(axes, [df_before, df_after], ['Before Collinearity Removal', 'After Collinearity Removal']):
        corr = df.select_dtypes(include=[np.number]).corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0,
                    square=True, linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_title(title, fontsize=12)
        ax.tick_params(axis='x', rotation=90, labelsize=7)
        ax.tick_params(axis='y', labelsize=7)

    plt.suptitle('Correlation Matrix: Before vs After', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = Path(output_dir) / '03_correlation_matrix.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_new_features(df: pd.DataFrame, output_dir: str):
    new_features = ['family_size', 'fare_per_person', 'age_group', 'title']
    existing = [c for c in new_features if c in df.columns]

    if 'age_group' in df.columns and df['age_group'].dtype.name != 'category':
        df = df.copy()

    n_cols = len(existing)
    if n_cols == 0:
        return
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    for ax, col in zip(axes, existing):
        if pd.api.types.is_numeric_dtype(df[col]):
            ax.hist(df[col].dropna(), bins=30, color='#4dabf7', edgecolor='white', alpha=0.8)
            ax.set_xlabel(col)
            ax.set_ylabel('Count')
        else:
            counts = df[col].value_counts()
            colors = plt.cm.Set2(np.linspace(0, 1, len(counts)))
            ax.bar(range(len(counts)), counts.values, color=colors)
            ax.set_xticks(range(len(counts)))
            ax.set_xticklabels(counts.index, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('Count')
        ax.set_title(col.replace('_', ' ').title())

    plt.suptitle('Engineered Features Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = Path(output_dir) / '04_engineered_features.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_pipeline_summary(initial_shape, final_shape, missing_before, missing_after,
                          n_features_engineered, output_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    metrics = [
        ('Dataset Shape', ['Initial', 'Final'], [initial_shape[1], final_shape[1]]),
        ('Missing Values', ['Before', 'After'], [missing_before, missing_after]),
        ('Features', ['Original', 'Engineered'], [initial_shape[1], n_features_engineered]),
    ]

    colors = [['#ff6b6b', '#51cf66'], ['#ff6b6b', '#51cf66'], ['#4dabf7', '#fcc419']]

    for ax, (title, labels, values), clrs in zip(axes, metrics, colors):
        bars = ax.bar(labels, values, color=clrs, width=0.5, edgecolor='white', linewidth=1.5)
        ax.set_title(title, fontsize=12, fontweight='bold')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                    str(val), ha='center', fontsize=11, fontweight='bold')

    plt.suptitle('Data Science Pipeline - Project Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    path = Path(output_dir) / '05_pipeline_summary.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def run_all(original: pd.DataFrame, cleaned: pd.DataFrame, initial_shape, output_dir: str = "figures"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    plot_missing_data(original, cleaned, output_dir)
    plot_outliers(original, cleaned, output_dir)
    plot_correlation_matrix(original, cleaned, output_dir)
    plot_new_features(cleaned, output_dir)
    plot_pipeline_summary(
        initial_shape=initial_shape,
        final_shape=cleaned.shape,
        missing_before=int(original.isnull().sum().sum()),
        missing_after=int(cleaned.isnull().sum().sum()),
        n_features_engineered=cleaned.shape[1] - initial_shape[1] + 1,
        output_dir=output_dir
    )

    print(f"\n  All visualizations saved to '{output_dir}/'")
    return output_dir
