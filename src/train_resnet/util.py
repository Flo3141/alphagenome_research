import numpy as np
import pandas as pd
import os

def load_embeddings(embeddings_path):
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


def save_loss_curves(loss_curve, val_loss_curve, fold_num, output_dir, loss_type):
    # Save fold loss curve to separate PNG
    try:
        import matplotlib
        matplotlib.use('Agg')  # Ensure non-interactive backend
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(loss_curve) + 1), loss_curve, label='Training Loss', color='royalblue', linewidth=2)
        plt.plot(range(1, len(val_loss_curve) + 1), val_loss_curve, label='Validation Loss', color='darkorange', linewidth=2)
        plt.title(f'ResNet Training & Validation Loss - Fold {fold_num} (Always Normalized)')
        plt.xlabel('Epoch')
        plt.ylabel(f'Loss ({loss_type.upper()})')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plot_path = os.path.join(output_dir, f"loss_curve_fold_{fold_num}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved fold loss curve plot to: {plot_path}")
    except Exception as e:
        print(f"Could not generate fold loss curve plot: {e}")

def save_loss_curves_as_csv(all_loss_curves, all_val_loss_curves, output_dir, loss_type):
    try:
        loss_data = {}
        max_len = max(max(len(c) for c in all_loss_curves), max(len(c) for c in all_val_loss_curves))
        for fold_idx in range(5):
            train_curve = all_loss_curves[fold_idx]
            val_curve = all_val_loss_curves[fold_idx]
            padded_train = train_curve + [np.nan] * (max_len - len(train_curve))
            padded_val = val_curve + [np.nan] * (max_len - len(val_curve))
            loss_data[f"fold_{fold_idx+1}_train"] = padded_train
            loss_data[f"fold_{fold_idx+1}_val"] = padded_val
        loss_data["epoch"] = list(range(1, max_len + 1))
        
        df_loss = pd.DataFrame(loss_data)
        cols = ['epoch']
        for i in range(5):
            cols.append(f"fold_{i+1}_train")
            cols.append(f"fold_{i+1}_val")
        df_loss = df_loss[cols]
        
        loss_csv_path = os.path.join(output_dir, "loss_curves.csv")
        df_loss.to_csv(loss_csv_path, index=False)
        print(f"Saved aggregated loss curves data to: {loss_csv_path}")
    except Exception as e:
        print(f"Could not save loss curves CSV: {e}")