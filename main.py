from src.pipeline import run

if __name__ == "__main__":
    df, report = run(
        data_path="data/titanic.csv",
        output_dir="output"
    )
