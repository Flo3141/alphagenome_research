import pandas as pd
import os
from sklearn.model_selection import train_test_split

def create_train_test_files():
    df_og = pd.read_csv(os.path.join(DATA_FOLDER,  "half_life_with_coords.csv"), sep=",")
    
    # We stratify for the half life, so that the distribution in the train and test set is roughly the same
    df_og['half_life_bins'] = pd.qcut(df_og['half_life'], q=NUM_BINS, labels=False, duplicates="drop")

    X_train, X_test = train_test_split(
        df_og,
        test_size=0.2,       # 20% test data
        random_state=42,
        shuffle=True,
        stratify=df_og['half_life_bins']
    )
    

    X_train = X_train.drop('half_life_bins', axis=1)
    X_test = X_test.drop('half_life_bins', axis=1)
    
    X_test.to_csv(os.path.join(DATA_FOLDER, f"half_life_with_coords_test.csv"), sep=",", index=False)

    X_train['half_life_bins'] = pd.qcut(X_train['half_life'], q=NUM_BINS, labels=False, duplicates="drop")
    X_train, X_val = train_test_split(
        X_train,
        test_size=0.25,       # 25% of remaining data is validation data => 0.8 * 0.25 = 20% of original data
        random_state=42,
        shuffle=True,
        stratify=X_train['half_life_bins']
    )
    X_train = X_train.drop('half_life_bins', axis=1)
    X_val = X_val.drop('half_life_bins', axis=1)
    X_train.to_csv(os.path.join(DATA_FOLDER, f"half_life_with_coords_train.csv"), sep=",", index=False)
    X_val.to_csv(os.path.join(DATA_FOLDER, f"half_life_with_coords_val.csv"), sep=",", index=False)

    print(f"Train set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Total size: {X_train.shape[0] + X_val.shape[0] + X_test.shape[0]}")


if __name__ == "__main__":
    DATA_FOLDER = os.environ.get("AG_DATA_FOLDER")
    NUM_BINS = 10
    create_train_test_files()