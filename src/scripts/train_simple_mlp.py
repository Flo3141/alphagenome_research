import argparse
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

def train_mlp_cv(
    embeddings_path="/beegfs/prj/RNA_NLP/AlphaGenome/embeddings/embeddings_train_val.npz",
    output_dir="/beegfs/prj/RNA_NLP/AlphaGenome/mlp_results",
    normalize_embeddings=True,
    random_state=42
):
    # Determine the directory for this configuration
    config_name = "with_normalization" if normalize_embeddings else "without_normalization"
    fold_output_dir = os.path.join(output_dir, config_name)
    os.makedirs(fold_output_dir, exist_ok=True)

    print(f"\n==========================================")
    print(f"Starting 5-Fold CV MLP (Embeddings Normalized: {normalize_embeddings})")
    print(f"Saving checkpoints to: {fold_output_dir}")
    print(f"==========================================")

    # Load embeddings and half_lives
    embeddings, _, half_lives = load_embeddings(embeddings_path)

    # Bin the targets to stratify continuous target values
    # We use 10 bins as done in create_train_val_test_splits.py
    print("Binning targets for stratification...")
    half_life_bins = pd.qcut(half_lives, q=10, labels=False, duplicates="drop")

    # Stratified K-Fold setup
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    fold_maes = []
    fold_mses = []
    fold_r2s = []
    all_loss_curves = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(embeddings, half_life_bins)):
        fold_num = fold + 1
        print(f"\n--- Fold {fold_num}/5 ---")
        X_train, X_val = embeddings[train_idx], embeddings[val_idx]
        y_train, y_val = half_lives[train_idx], half_lives[val_idx]

        # Normalize features if flag is set
        scaler = None
        if normalize_embeddings:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

        # Simple MLP Regressor: 100 epochs (max_iter=100)
        # Hidden layer sizes: 256 then 128
        # solver='adam', activation='relu'
        model = MLPRegressor(
            hidden_layer_sizes=(256, 128),
            activation='relu',
            solver='adam',
            max_iter=100,
            random_state=random_state,
            verbose=False
        )

        print(f"Training model on {len(X_train)} samples...")
        model.fit(X_train, y_train)

        # Save the fold checkpoint
        checkpoint_path = os.path.join(fold_output_dir, f"mlp_model_fold_{fold_num}.pkl")
        checkpoint_data = {
            "model": model,
            "scaler": scaler,
            "loss_curve": model.loss_curve_
        }
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        print(f"Saved checkpoint to: {checkpoint_path}")

        # Save fold loss curve to separate PNG
        try:
            import matplotlib
            matplotlib.use('Agg')  # Ensure non-interactive backend
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(8, 5))
            plt.plot(range(1, len(model.loss_curve_) + 1), model.loss_curve_, label='Training Loss', color='darkorange', linewidth=2)
            plt.title(f'MLP Training Loss Curve - Fold {fold_num} (Normalized: {normalize_embeddings})')
            plt.xlabel('Epoch')
            plt.ylabel('Loss (MSE)')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            
            plot_path = os.path.join(fold_output_dir, f"loss_curve_fold_{fold_num}.png")
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"Saved fold loss curve plot to: {plot_path}")
        except Exception as e:
            print(f"Could not generate fold loss curve plot: {e}")

        # Predict
        y_pred = model.predict(X_val)

        # Metrics
        mse = mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        print(f"Fold {fold_num} results: MAE = {mae:.4f}, MSE = {mse:.4f}, R^2 = {r2:.4f}")
        fold_maes.append(mae)
        fold_mses.append(mse)
        fold_r2s.append(r2)
        all_loss_curves.append(model.loss_curve_)

    # Save aggregated loss curves as CSV
    try:
        loss_data = {}
        max_len = max(len(curve) for curve in all_loss_curves)
        for fold_idx, curve in enumerate(all_loss_curves):
            # Pad with NaN if any fold finished early
            padded = curve + [np.nan] * (max_len - len(curve))
            loss_data[f"fold_{fold_idx+1}"] = padded
        loss_data["epoch"] = list(range(1, max_len + 1))
        
        df_loss = pd.DataFrame(loss_data)
        cols = ['epoch'] + [f"fold_{i+1}" for i in range(5)]
        df_loss = df_loss[cols]
        
        loss_csv_path = os.path.join(fold_output_dir, "loss_curves.csv")
        df_loss.to_csv(loss_csv_path, index=False)
        print(f"Saved aggregated loss curves data to: {loss_csv_path}")
    except Exception as e:
        print(f"Could not save loss curves CSV: {e}")

    # Calculate average scores
    avg_mae = np.mean(fold_maes)
    avg_mse = np.mean(fold_mses)
    avg_r2 = np.mean(fold_r2s)

    print(f"\nSummary for Normalized={normalize_embeddings}:")
    print(f"  Avg MAE: {avg_mae:.4f}")
    print(f"  Avg MSE: {avg_mse:.4f}")
    print(f"  Avg R^2: {avg_r2:.4f}")

    return {
        'mae': avg_mae,
        'mse': avg_mse,
        'r2': avg_r2
    }

def run_final_test_evaluation(
    test_path="/beegfs/prj/RNA_NLP/AlphaGenome/embeddings/embeddings_test.npz",
    best_config_dir="/beegfs/prj/RNA_NLP/AlphaGenome/mlp_results/with_normalization"
):
    print(f"\n==========================================")
    print(f"Running Final Test Evaluation from: {best_config_dir}")
    print(f"==========================================")
    
    # Load test set
    X_test, _, y_test = load_embeddings(test_path)

    # Accumulate predictions across all 5 folds
    all_preds = []

    for fold_idx in range(5):
        fold_num = fold_idx + 1
        model_path = os.path.join(best_config_dir, f"mlp_model_fold_{fold_num}.pkl")
        print(f"Loading checkpoint for fold {fold_num} from {model_path}...")
        
        with open(model_path, 'rb') as f:
            checkpoint = pickle.load(f)
            
        model = checkpoint["model"]
        scaler = checkpoint["scaler"]

        # Apply fold scaler if present
        if scaler is not None:
            X_test_fold = scaler.transform(X_test)
        else:
            X_test_fold = X_test

        # Predict
        y_pred_fold = model.predict(X_test_fold)
        all_preds.append(y_pred_fold)

    # Average predictions of the 5 models
    y_pred_ensemble = np.mean(all_preds, axis=0)

    # Metrics
    mse = mean_squared_error(y_test, y_pred_ensemble)
    mae = mean_absolute_error(y_test, y_pred_ensemble)
    r2 = r2_score(y_test, y_pred_ensemble)

    print(f"\n==========================================")
    print(f"FINAL TEST SET RESULTS (5-Fold Ensemble):")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R^2: {r2:.4f}")
    print(f"==========================================")

def run_training_experiment(
    train_val_path="/beegfs/prj/RNA_NLP/AlphaGenome/embeddings/embeddings_train_val.npz",
    test_path="/beegfs/prj/RNA_NLP/AlphaGenome/embeddings/embeddings_test.npz",
    output_dir="/beegfs/prj/RNA_NLP/AlphaGenome/mlp_results"
):
    # Run CV with normalization (saves fold checkpoints under output_dir/with_normalization)
    metrics_norm = train_mlp_cv(
        embeddings_path=train_val_path,
        output_dir=output_dir,
        normalize_embeddings=True
    )
    
    # Run CV without normalization (saves fold checkpoints under output_dir/without_normalization)
    metrics_raw = train_mlp_cv(
        embeddings_path=train_val_path,
        output_dir=output_dir,
        normalize_embeddings=False
    )
    
    # Compare
    print("\n" + "="*50)
    print(" COMPARISON: MLP ON ALPHAGENOME EMBEDDINGS (5-Fold CV)")
    print("="*50)
    print(f"Metric      | With Normalization | Without Normalization")
    print(f"------------|--------------------|----------------------")
    print(f"Avg MAE     | {metrics_norm['mae']:.5f}            | {metrics_raw['mae']:.5f}")
    print(f"Avg MSE     | {metrics_norm['mse']:.5f}            | {metrics_raw['mse']:.5f}")
    print(f"Avg R^2     | {metrics_norm['r2']:.5f}            | {metrics_raw['r2']:.5f}")
    print("="*50)

    # Determine best configuration based on Avg MAE
    best_config = "with_normalization" if metrics_norm['mae'] < metrics_raw['mae'] else "without_normalization"
    best_config_dir = os.path.join(output_dir, best_config)
    print(f"\nBest configuration selected based on CV MAE: {best_config}")

    # Run final test set evaluation using the loaded ensemble models
    run_final_test_evaluation(
        test_path=test_path,
        best_config_dir=best_config_dir
    )

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset and train simple MLP on AlphaGenome embeddings.")
    parser.add_argument(
        "--split",
        action="store_true",
        help="Split original embeddings.npz into train_val and test sets."
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the simple MLP on the train_val embeddings using 5-fold CV and evaluate on the test set."
    )
    parser.add_argument(
        "--embeddings_path",
        type=str,
        default="/beegfs/prj/RNA_NLP/AlphaGenome/embeddings/embeddings_train_val.npz",
        help="Path to embeddings_train_val.npz file for training."
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="/beegfs/prj/RNA_NLP/AlphaGenome/embeddings/embeddings_test.npz",
        help="Path to embeddings_test.npz file for final evaluation."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/beegfs/prj/RNA_NLP/AlphaGenome/mlp_results",
        help="Directory to save the final models, loss CSVs and plot PNGs."
    )
    args = parser.parse_args()

    if args.split:
        split_embeddings()
    elif args.train:
        run_training_experiment(
            train_val_path=args.embeddings_path,
            test_path=args.test_path,
            output_dir=args.output_dir
        )
    else:
        # If neither flag is set, check if train_val split exists, and run training.
        # Otherwise fall back to printing help.
        if os.path.exists(args.embeddings_path):
            print(f"Found {args.embeddings_path}. Running training experiment...")
            run_training_experiment(
                train_val_path=args.embeddings_path,
                test_path=args.test_path,
                output_dir=args.output_dir
            )
        else:
            parser.print_help()

if __name__ == "__main__":
    main()