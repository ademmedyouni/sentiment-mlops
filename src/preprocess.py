import os
import re
import pandas as pd


def clean(text):
    text = "" if pd.isna(text) else str(text)
    text = re.sub(r"<.*?>", "", text)
    return text.lower().strip()


def main():
    raw_path = "data/raw/train.csv"
    output_path = "data/processed/train_clean.csv"

    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"{raw_path} not found. Run download_data.py first."
        )
    os.makedirs("data/processed", exist_ok=True)
    train_df = pd.read_csv(raw_path)
    if "text" not in train_df.columns or "label" not in train_df.columns:
        raise ValueError("train.csv must contain 'text' and 'label' columns")
    train_df["text"] = train_df["text"].apply(clean)
    train_df.to_csv(output_path, index=False)
    print(f"Done: saved {len(train_df)} rows to {output_path}")


if __name__ == "__main__":
    main()