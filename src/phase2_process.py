import pandas as pd
import numpy as np


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    print("\n[PROCESS] Categorical Encoding:")

    binary_cols = ['sex', 'adult_male', 'alone']
    for col in binary_cols:
        if col in df.columns:
            unique_vals = df[col].dropna().unique()
            if pd.api.types.is_string_dtype(df[col]):
                mapping = {val: i for i, val in enumerate(sorted(unique_vals))}
                df[col] = df[col].map(mapping)
            elif pd.api.types.is_bool_dtype(df[col]):
                df[col] = df[col].astype(int)
            else:
                df[col] = df[col].astype(int)
                mapping = {}
            print(f"  Binary encoded '{col}' -> {dict(zip(unique_vals, range(len(unique_vals))))}")

    multi_cat_cols = []
    for col in df.columns:
        if col in ['alive', 'class', 'survived']:
            continue
        if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            n_unique = df[col].nunique()
            if 2 < n_unique < 10:
                multi_cat_cols.append(col)

    for col in multi_cat_cols:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False, dtype=int)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(columns=[col])
        print(f"  One-hot encoded '{col}' -> {dummies.columns.tolist()}")

    if 'class' in df.columns:
        class_map = {'First': 1, 'Second': 2, 'Third': 3}
        df['class_encoded'] = df['class'].map(class_map)
        print(f"  Ordinal encoded 'class' -> class_encoded")

    if 'alive' in df.columns:
        df = df.drop(columns=['alive'])

    return df


def eradicate_multicollinearity(df: pd.DataFrame, target_col: str = 'survived', threshold: float = 0.80) -> pd.DataFrame:
    df = df.copy()
    print("\n[PROCESS] Multicollinearity Eradication:")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    corr_matrix = df[numeric_cols].corr()

    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    high_corr_pairs = []
    for col in upper_tri.columns:
        for idx in upper_tri.index:
            val = upper_tri.loc[idx, col]
            if abs(val) > threshold and not pd.isna(val):
                high_corr_pairs.append((idx, col, abs(val)))

    if not high_corr_pairs:
        print("  No highly correlated pairs found above threshold.")
        return df

    print(f"  Found {len(high_corr_pairs)} highly correlated pair(s) (|r| > {threshold}):")
    to_drop = set()
    for c1, c2, val in high_corr_pairs:
        corr_c1 = abs(df[c1].corr(df[target_col])) if target_col in df.columns else 0
        corr_c2 = abs(df[c2].corr(df[target_col])) if target_col in df.columns else 0
        weaker = c2 if corr_c1 >= corr_c2 else c1
        stronger = c1 if corr_c1 >= corr_c2 else c2
        print(f"    {c1} & {c2}: r={val:.3f} -> drop '{weaker}' (weaker target corr)")
        to_drop.add(weaker)

    df = df.drop(columns=list(to_drop), errors='ignore')
    print(f"  Dropped columns: {list(to_drop)}")

    return df


def run(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("PHASE 2: VECTORIZED COMPUTATION ENGINE")
    print("=" * 60)

    df = encode_categorical(df)
    df = eradicate_multicollinearity(df)

    print(f"\n[PROCESS] Phase 2 complete.")
    print(f"  Shape after process phase: {df.shape}")
    return df
