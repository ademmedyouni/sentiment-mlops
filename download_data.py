import os


source_train = "data/raw/train.csv"
source_test = "data/raw/test.csv"

if os.path.exists(source_train) and os.path.exists(source_test):
    print("Local raw data already exists:")
    print(f"  {source_train}")
    print(f"  {source_test}")
else:
    raise FileNotFoundError(
        "Local raw CSV files are missing. Put your dataset in data/raw/train.csv and data/raw/test.csv."
    )
