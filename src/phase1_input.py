import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


def analyze_missingness(df: pd.DataFrame) -> pd.DataFrame:
    missing = pd.DataFrame({
        'column': df.columns,
        'dtype': df.dtypes.values,
        'missing_count': df.isnull().sum().values,
        'missing_pct': (df.isnull().sum() / len(df) * 100).values
    })
    missing = missing[missing['missing_count'] > 0].sort_values('missing_pct', ascending=False)
    print("\n[INPUT] Missing Data Analysis:")
    print(missing.to_string(index=False))
    return missing


def impute_by_strategy(df: pd.DataFrame, col: str, strategy: str) -> pd.DataFrame:
    df = df.copy()
    missing_pct = df[col].isnull().mean() * 100

    if strategy == 'drop':
        df = df.dropna(subset=[col])
        print(f"  Dropped rows with missing '{col}' ({missing_pct:.1f}% missing)")

    elif strategy == 'median':
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"  Imputed '{col}' with median ({median_val:.2f})")

    elif strategy == 'mean':
        mean_val = df[col].mean()
        df[col] = df[col].fillna(mean_val)
        print(f"  Imputed '{col}' with mean ({mean_val:.2f})")

    elif strategy == 'group_median':
        group_col = 'pclass'
        group_medians = df.groupby(group_col)[col].transform('median')
        df[col] = df[col].fillna(group_medians)
        print(f"  Imputed '{col}' with median grouped by '{group_col}'")

    elif strategy == 'mode':
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
        print(f"  Imputed '{col}' with mode ({mode_val})")

    return df


def knn_impute(df: pd.DataFrame, cols_to_impute: list, n_neighbors: int = 5) -> pd.DataFrame:
    df = df.copy()
    print(f"  Applying KNN imputation (k={n_neighbors}) on {cols_to_impute}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    impute_cols = [c for c in cols_to_impute if c in numeric_cols]

    if impute_cols:
        scaler = StandardScaler()
        imputer = KNNImputer(n_neighbors=n_neighbors)

        X = df[numeric_cols].copy()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=numeric_cols,
            index=X.index
        )

        X_imputed = pd.DataFrame(
            imputer.fit_transform(X_scaled),
            columns=numeric_cols,
            index=X.index
        )

        df[impute_cols] = X_imputed[impute_cols]
        print(f"  KNN imputation complete for: {impute_cols}")

    return df


def apply_missing_data_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    missing = analyze_missingness(df)

    for _, row in missing.iterrows():
        col = row['column']
        pct = row['missing_pct']

        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        is_categorical = pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col])

        if pct < 5:
            if is_categorical:
                df = impute_by_strategy(df, col, 'mode')
            else:
                df = impute_by_strategy(df, col, 'median')
        elif pct < 20:
            if is_categorical:
                df = impute_by_strategy(df, col, 'mode')
            elif is_numeric:
                df = impute_by_strategy(df, col, 'group_median')
        else:
            if is_numeric:
                df = knn_impute(df, [col], n_neighbors=5)
            else:
                print(f"  Dropped column '{col}' ({pct:.1f}% missing - too high for imputation)")
                df = df.drop(columns=[col])

    return df


def detect_outliers_iqr(df: pd.DataFrame, col: str) -> pd.Series:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    is_outlier = (df[col] < lower) | (df[col] > upper)
    return is_outlier


def detect_outliers_zscore(df: pd.DataFrame, col: str, threshold: float = 3.0) -> pd.Series:
    z = np.abs((df[col] - df[col].mean()) / df[col].std())
    return z > threshold


def neutralize_outliers_winsorize(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_before = ((df[col] < lower) | (df[col] > upper)).sum()
    df[col] = df[col].clip(lower=lower, upper=upper)
    print(f"  Winsorized '{col}': capped {n_before} outliers at [{lower:.2f}, {upper:.2f}]")
    return df


def neutralize_outliers_drop(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    mask = (df[col] >= lower) & (df[col] <= upper)
    n_dropped = (~mask).sum()
    df = df[mask]
    print(f"  Dropped {n_dropped} rows with outliers in '{col}'")
    return df


def apply_outlier_treatment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    print("\n[INPUT] Outlier Detection & Treatment:")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ['survived', 'sibsp', 'parch']
    target_cols = [c for c in numeric_cols if c not in exclude]

    for col in target_cols:
        outlier_mask = detect_outliers_iqr(df, col)
        n_outliers = outlier_mask.sum()
        if n_outliers > 0:
            print(f"  '{col}': {n_outliers} outliers ({n_outliers/len(df)*100:.1f}%)")
            df = neutralize_outliers_winsorize(df, col)

    return df


def run(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 60)
    print("PHASE 1: SECURING INPUT FIDELITY")
    print("=" * 60)

    df = apply_missing_data_matrix(df)
    df = apply_outlier_treatment(df)

    print("\n[INPUT] Phase 1 complete.")
    print(f"  Shape after input phase: {df.shape}")
    return df
