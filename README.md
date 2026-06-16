# 📊 Titanic EDA & Feature Engineering

**Production-Grade Data Pipeline — Input → Process → Output Architecture**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![Pandas](https://img.shields.io/badge/Pandas-3.0-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![NumPy](https://img.shields.io/badge/NumPy-2.4-013243?logo=numpy&logoColor=white)](https://numpy.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.8-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8-11557C?logo=python&logoColor=white)](https://matplotlib.org)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.13-3776AB?logo=python&logoColor=white)](https://seaborn.pydata.org)

---

## 📋 Overview

A **production-grade data engineering pipeline** that transforms raw, chaotic data into a mathematically clean, ML-ready dataset. Built following the **Input-Process-Output (IPO)** blueprint — the same architecture used in enterprise feature stores and MLOps systems.

The pipeline ingests the Titanic dataset, handles missing values via statistical imputation, neutralizes outliers using IQR-based winsorization, engineers new predictive features, encodes categorical variables, eradicates multicollinearity, and generates formal data contracts for downstream consumers.

Built for the **DecodeLabs Data Science** industrial training program.

---

## ✨ Features

- **🧹 Intelligent Missing Data Handling** — Decision matrix-based imputation (median, group-wise, mode, KNN) based on missingness proportion
- **📐 IQR Outlier Treatment** — Non-parametric boundary detection with winsorization (preserves row count)
- **🧠 6+ Engineered Features** — `family_size`, `fare_per_person`, `age_group`, `title`, interaction terms
- **🏷️ Categorical Encoding** — One-Hot, Binary, and Ordinal encoding with no synthetic spatial hierarchy
- **🔗 Multicollinearity Eradication** — Pearson correlation with target-aware feature selection (r > 0.80 threshold)
- **📋 Formal Data Contracts** — Schema validation, statistical boundary checks, point-in-time correctness verification
- **📊 Pipeline Visualizations** — Before/after comparison plots for missing data, outliers, correlations, and engineered features

---

## 🗂️ Project Structure

```
titanic-eda-feature-engineering/
├── 📁 data/
│   └── titanic.csv                  # Raw Titanic dataset (891 rows, 15 cols)
│
├── 📁 src/
│   ├── phase1_input.py              # 🔵 INPUT: Missing imputation + outlier treatment
│   ├── feature_engineering.py        # 🟡 Feature creation (6+ new features)
│   ├── phase2_process.py            # 🟢 PROCESS: Encoding + collinearity eradication
│   ├── phase3_output.py             # 🟠 OUTPUT: Schema contracts + validation
│   ├── pipeline.py                  # 🔄 Pipeline orchestrator
│   └── visualize.py                 # 📈 Visualization generator
│
├── 📁 output/                       # Pipeline artifacts
│   ├── cleaned_dataset.csv          # Processed data (0 missing values)
│   ├── schema_report.csv            # Per-column dtype & missingness
│   ├── data_contract.csv            # Formal feature contracts
│   └── boundary_warnings.csv        # IQR boundary violations
│
├── 📁 figures/                      # Generated visualizations
│   ├── 01_missing_data.png
│   ├── 02_outlier_treatment.png
│   ├── 03_correlation_matrix.png
│   ├── 04_engineered_features.png
│   └── 05_pipeline_summary.png
│
├── main.py                          # Entry point
├── run.bat                          # VS Code launcher
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- pip

### Setup

```bash
git clone https://github.com/abdur-codes/Titanic-EDA-Feature-Engineering.git
cd Titanic-EDA-Feature-Engineering

# Run the pipeline
python main.py
```

Or double-click `run.bat` to open in VS Code, then press **`Ctrl+Shift+B`**.

---

## 🧠 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                       INPUT-PROCESS-OUTPUT (IPO)                    │
├───────────┬─────────────────────────┬───────────────────────────────┤
│  PHASE 1  │      PHASE 2            │         PHASE 3               │
│  INPUT    │      PROCESS            │         OUTPUT                │
├───────────┼─────────────────────────┼───────────────────────────────┤
│           │                         │                               │
│  Missing  │   Binary / OHE /        │   Schema Validation           │
│  Data     │   Ordinal Encoding      │   (dtypes, boundaries)        │
│  Matrix   │                         │                               │
│           │   Multicollinearity     │   Data Contract               │
│  IQR      │   Eradication           │   Generation                  │
│  Outlier  │   (target-aware         │                               │
│  Caps     │    Pearson, r>0.80)     │   Point-in-Time               │
│           │                         │   Leakage Check               │
│  KNN      │   Vectorized Ops        │                               │
│  Impute   │   (no loops)            │   Statistical                 │
│           │                         │   Boundary Checks             │
└───────────┴─────────────────────────┴───────────────────────────────┘
```

---

## 📊 Results

| Metric | Before | After |
|--------|--------|-------|
| **Dataset Shape** | 891 × 15 | 891 × **18** |
| **Missing Values** | 869 | **0** ✅ |
| **Features Engineered** | — | **6+** |
| **Collinear Features Removed** | — | **14** |
| **Missing %** | 6.5% | **0%** |

### Engineered Features

| Feature | Formula | Type |
|---------|---------|------|
| `family_size` | `sibsp + parch + 1` | Numeric (count) |
| `is_alone` | `family_size == 1` | Binary indicator |
| `fare_per_person` | `fare / family_size` | Normalized numeric |
| `age_group` | Cut into [Child, Teen, Adult, Middle_Aged, Senior] | Ordinal |
| `title` | Extracted from `who` column [Mr, Mrs, Master] | Categorical |
| `pclass_sex_interaction` | `pclass × sex_numeric` | Cross-feature interaction |

---

## 📈 Visualizations

| Figure | Preview | Description |
|--------|---------|-------------|
| **Missing Data** | ![Missing Data](figures/01_missing_data.png) | Before/after comparison of missing values per column |
| **Outlier Treatment** | ![Outliers](figures/02_outlier_treatment.png) | Box plots before and after winsorization |
| **Correlation Matrix** | ![Correlation](figures/03_correlation_matrix.png) | Correlation heatmap before/after collinearity removal |
| **Engineered Features** | ![Features](figures/04_engineered_features.png) | Distribution of new features across passengers |
| **Pipeline Summary** | ![Summary](figures/05_pipeline_summary.png) | Overall pipeline impact dashboard |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.12 |
| **Data Manipulation** | Pandas 3.0, NumPy 2.4 |
| **Statistical Methods** | IQR, Z-Score, Winsorization, KNN Imputation (scikit-learn) |
| **Encoding** | One-Hot, Binary, Ordinal |
| **Validation** | Schema contracts, statistical boundary checks, leakage detection |
| **Visualization** | Matplotlib 3.8, Seaborn 0.13 |
| **Containerization** | Git (version control) |

---

## 🔬 Key Techniques

### Missing Data Decision Matrix
```
Missingness < 5%   → Drop rows (dropna) or Mode imputation
5% ≤ Missing < 20% → Group-wise Median (numeric) / Mode (categorical)
Missing ≥ 20%      → KNN Imputation (k=5, numeric) / Drop column (categorical)
```

### Outlier Treatment
- **Method:** Interquartile Range (IQR) — Q1 − 1.5×IQR to Q3 + 1.5×IQR
- **Strategy:** Winsorization (`numpy.clip`) — preserves row count for temporal integrity
- **Impact:** 26 age outliers + 116 fare outliers capped at statistical boundaries

### Multicollinearity Eradication
1. Build absolute Pearson correlation matrix
2. Extract upper triangle (r > 0.80 threshold)
3. For each collinear pair, compare correlation with target variable
4. Drop the feature with weaker target correlation

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install pandas numpy scikit-learn matplotlib seaborn` |
| `FileNotFoundError: data/titanic.csv` | Ensure you're in the project root directory |
| Unicode errors in terminal | Use `python main.py` (not `python3 main.py`) |
| Figures not generating | Check `matplotlib` and `seaborn` are installed |

---

## 🤝 Connect

[![GitHub](https://img.shields.io/badge/GitHub-abdur--codes-181717?logo=github)](https://github.com/abdur-codes)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin)](https://www.linkedin.com/in/abdur-rehman-581305358)

**If you find this project useful, consider giving it a ⭐!**

---

**DecodeLabs** — Data Science Industrial Training — Batch 2026
