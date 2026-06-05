import argparse
import os
import numpy as np
import pandas as pd

def load_embeddings(embeddings_path="/beegfs/prj/RNA_NLP/AlphaGenome/embeddings/embeddings.npz"):
    data = np.load(embeddings_path)
    embeddings = data["embeddings"]
    transcript_ids = data["ensembl_transcript_ids"]
    half_lives = data["half_lives"]

    print(f"Loaded embeddings from: {embeddings_path}")
    print("Embeddings Shape:", embeddings.shape)
    print("Transcript IDs Shape:", transcript_ids.shape)
    print("Half Lives Shape:", half_lives.shape)

    if len(transcript_ids) > 0:
        print("First transcript ID:", transcript_ids[0])
        print("First half life:", half_lives[0])
        print("First embedding sample (first 3 values):", embeddings[0][:3])
    
    return embeddings, transcript_ids, half_lives

def split_embeddings():
    all_embeddings_path="/beegfs/prj/RNA_NLP/AlphaGenome/embeddings/embeddings.npz"
    test_csv_path="/beegfs/prj/RNA_NLP/AlphaGenome/data/half_life_with_coords_test.csv"

    print(f"Reading test transcript IDs from {test_csv_path}...")
    test_df = pd.read_csv(test_csv_path)
    test_ids = set(test_df["ensembl_transcript_id"].dropna().astype(str).unique())
    print(f"Found {len(test_ids)} unique transcript IDs in test CSV.")

    print(f"Loading original embeddings from {all_embeddings_path}...")
    data = np.load(all_embeddings_path)
    embeddings = data["embeddings"]
    transcript_ids = data["ensembl_transcript_ids"]
    half_lives = data["half_lives"]

    # Ensure transcript IDs are compared as strings
    transcript_ids_str = transcript_ids.astype(str)

    # Generate boolean masks
    test_mask = np.isin(transcript_ids_str, list(test_ids))
    train_val_mask = ~test_mask

    print(f"Total original samples: {len(transcript_ids)}")
    print(f"Test set samples: {np.sum(test_mask)}")
    print(f"Train/Val set samples: {np.sum(train_val_mask)}")

    # Define output file paths in the same directory as embeddings_path
    base_dir = os.path.dirname(all_embeddings_path)
    test_output_path = os.path.join(base_dir, "embeddings_test.npz")
    train_val_output_path = os.path.join(base_dir, "embeddings_train_val.npz")

    # Save test embeddings
    np.savez(
        test_output_path,
        embeddings=embeddings[test_mask],
        ensembl_transcript_ids=transcript_ids[test_mask],
        half_lives=half_lives[test_mask]
    )
    print(f"Successfully saved test embeddings to {test_output_path}")

    # Save train/val embeddings
    np.savez(
        train_val_output_path,
        embeddings=embeddings[train_val_mask],
        ensembl_transcript_ids=transcript_ids[train_val_mask],
        half_lives=half_lives[train_val_mask]
    )
    print(f"Successfully saved train/val embeddings to {train_val_output_path}")

if __name__ == "__main__":
    split_embeddings()