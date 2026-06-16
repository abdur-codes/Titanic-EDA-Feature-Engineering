import pandas as pd
import numpy as np


def validate_schema(df: pd.DataFrame) -> dict:
    print("\n[OUTPUT] Schema Validation:")

    schema_report = {
        'shape': df.shape,
        'total_cells': df.shape[0] * df.shape[1],
        'missing_cells': int(df.isnull().sum().sum()),
        'missing_pct': round(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100, 4),
        'dtypes': {},
        'column_report': []
    }

    print(f"  Shape: {df.shape}")
    print(f"  Missing cells: {schema_report['missing_cells']} ({schema_report['missing_pct']}%)")

    for col in df.columns:
        col_info = {
            'column': col,
            'dtype': str(df[col].dtype),
            'missing': int(df[col].isnull().sum()),
            'missing_pct': round(df[col].isnull().mean() * 100, 2),
            'memory_kb': round(df[col].memory_usage(deep=True) / 1024, 2)
        }

        if df[col].dtype in ['int64', 'float64']:
            col_info['min'] = round(float(df[col].min()), 4)
            col_info['max'] = round(float(df[col].max()), 4)
            col_info['mean'] = round(float(df[col].mean()), 4)
            col_info['std'] = round(float(df[col].std()), 4)

        schema_report['column_report'].append(col_info)

        status = "OK" if col_info['missing'] == 0 else "WARN"
        print(f"  [{status}] {col}: {col_info['dtype']}, missing={col_info['missing']}")

    return schema_report


def check_statistical_boundaries(df: pd.DataFrame) -> list:
    print("\n[OUTPUT] Statistical Boundary Checks:")
    warnings = []

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        violations = ((df[col] < lower) | (df[col] > upper)).sum()
        if violations > 0:
            msg = f"  WARN: '{col}' has {violations} values outside IQR bounds"
            warnings.append(msg)
            print(msg)

    if not warnings:
        print("  All features within statistical boundaries (no IQR violations)")

    return warnings


def generate_data_contract(df: pd.DataFrame) -> dict:
    contract = {
        'version': '1.0.0',
        'entity': 'passenger',
        'record_count': len(df),
        'feature_count': len(df.columns),
        'features': {}
    }

    for col in df.columns:
        feat = {
            'dtype': str(df[col].dtype),
            'nullable': bool(df[col].isnull().any()),
            'unique_values': int(df[col].nunique()),
        }

        if df[col].dtype in ['int64', 'float64']:
            feat['bounds'] = {
                'min': round(float(df[col].min()), 4),
                'max': round(float(df[col].max()), 4)
            }

        contract['features'][col] = feat

    print("\n[OUTPUT] Data Contract Generated:")
    print(f"  Version: {contract['version']}")
    print(f"  Records: {contract['record_count']}")
    print(f"  Features: {contract['feature_count']}")

    return contract


def verify_no_leakage(df: pd.DataFrame, target_col: str = 'survived') -> bool:
    print("\n[OUTPUT] Point-in-Time Correctness Check:")
    future_leak_cols = ['alive']
    leaked = [c for c in future_leak_cols if c in df.columns]
    if leaked:
        print(f"  WARN: Potential leakage columns found: {leaked}")
        return False
    else:
        print(f"  OK: No future-information leakage detected")
        return True


def run(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 60)
    print("PHASE 3: STRUCTURAL CONTRACTS & SCALING")
    print("=" * 60)

    schema = validate_schema(df)
    boundaries = check_statistical_boundaries(df)
    contract = generate_data_contract(df)
    leakage = verify_no_leakage(df)

    final_report = {
        'schema': schema,
        'boundary_warnings': boundaries,
        'contract': contract,
        'no_leakage': leakage
    }

    missing_remaining = df.isnull().sum().sum()
    print(f"\n[OUTPUT] Phase 3 complete.")
    print(f"  Remaining missing values: {missing_remaining}")
    print(f"  Final shape: {df.shape}")

    return final_report
