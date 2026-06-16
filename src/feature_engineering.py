import pandas as pd
import numpy as np


def create_family_size(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'sibsp' in df.columns and 'parch' in df.columns:
        df['family_size'] = df['sibsp'] + df['parch'] + 1
        df['is_alone'] = (df['family_size'] == 1).astype(int)
        print("  Feature: 'family_size' (sibsp + parch + 1)")
        print("  Feature: 'is_alone' (family_size == 1)")
    return df


def create_fare_per_person(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'fare' in df.columns and 'family_size' in df.columns:
        df['fare_per_person'] = df['fare'] / df['family_size']
        df['fare_per_person'] = df['fare_per_person'].replace([np.inf, -np.inf], np.nan).fillna(0)
        print("  Feature: 'fare_per_person' (fare / family_size)")
    return df


def create_age_group(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'age' in df.columns:
        bins = [0, 12, 18, 35, 60, 100]
        labels = ['Child', 'Teen', 'Adult', 'Middle_Aged', 'Senior']
        df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)
        print("  Feature: 'age_group' (Child/Teen/Adult/Middle_Aged/Senior)")
    return df


def create_title(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'who' in df.columns:
        title_map = {'man': 'Mr', 'woman': 'Mrs', 'child': 'Master'}
        df['title'] = df['who'].map(title_map).fillna('Unknown')
        print("  Feature: 'title' (Mr/Mrs/Master based on who)")
    return df


def create_deck_group(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'deck' in df.columns:
        df['deck_group'] = df['deck'].astype(str).str[0]
        df['deck_group'] = df['deck_group'].replace('n', 'Unknown')
        print("  Feature: 'deck_group' (first letter of deck)")
    return df


def create_interaction_pclass_sex(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'pclass' in df.columns and 'sex' in df.columns:
        sex_map = {'male': 0, 'female': 1}
        sex_numeric = df['sex'].map(sex_map).fillna(0).astype(int)
        df['pclass_sex_interaction'] = df['pclass'] * sex_numeric
        print("  Feature: 'pclass_sex_interaction' (pclass * sex_numeric)")
    return df


def run(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    df = create_family_size(df)
    df = create_fare_per_person(df)
    df = create_age_group(df)
    df = create_title(df)
    df = create_deck_group(df)
    df = create_interaction_pclass_sex(df)

    print(f"\n[FE] Created 6+ new features.")
    print(f"  Shape after feature engineering: {df.shape}")
    return df
