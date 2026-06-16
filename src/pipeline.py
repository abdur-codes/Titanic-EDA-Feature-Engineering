import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import phase1_input, phase2_process, phase3_output, feature_engineering, visualize


def load_data(path: str) -> pd.DataFrame:
    print("=" * 60)
    print("DATA LOADING")
    print("=" * 60)
    if path.endswith('.csv'):
        df = pd.read_csv(path)
    elif path.endswith('.parquet'):
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported format: {path}")
    print(f"  Loaded: {path}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    return df


def save_artifacts(df: pd.DataFrame, report: dict, output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df.to_csv(f"{output_dir}/cleaned_dataset.csv", index=False)
    print(f"\n  Saved cleaned dataset: {output_dir}/cleaned_dataset.csv")

    report_df = pd.DataFrame(report['schema']['column_report'])
    report_df.to_csv(f"{output_dir}/schema_report.csv", index=False)
    print(f"  Saved schema report: {output_dir}/schema_report.csv")

    contract_df = pd.DataFrame.from_dict(
        report['contract']['features'], orient='index'
    )
    contract_df.to_csv(f"{output_dir}/data_contract.csv")
    print(f"  Saved data contract: {output_dir}/data_contract.csv")

    if report['boundary_warnings']:
        warnings_df = pd.DataFrame({
            'warning': report['boundary_warnings']
        })
        warnings_df.to_csv(f"{output_dir}/boundary_warnings.csv", index=False)
        print(f"  Saved boundary warnings: {output_dir}/boundary_warnings.csv")


def run(data_path: str, output_dir: str = "output", figures_dir: str = "figures"):
    df_original = load_data(data_path)
    initial_shape = df_original.shape

    df = phase1_input.run(df_original.copy())
    df = feature_engineering.run(df)
    df = phase2_process.run(df)
    report = phase3_output.run(df)

    save_artifacts(df, report, output_dir)
    visualize.run_all(df_original, df, initial_shape, figures_dir)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Initial shape: {initial_shape}")
    print(f"  Final shape:   {df.shape}")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    print(f"  Features engineered: 6+")
    print(f"  Output directory: {output_dir}/")
    print(f"  Figures: {figures_dir}/")

    return df, report
