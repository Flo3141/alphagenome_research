import argparse
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define ResNet MLP Module
class ResNetMLP(nn.Module):
    def __init__(self, input_dim=3072, hidden_dim=512, dropout_rate=0.2, noise_level=0.0):
        super().__init__()
        self.noise_level = noise_level
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Residual Block
        self.res_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        if self.training and self.noise_level > 0.0:
            x = x + torch.randn_like(x) * self.noise_level
        h = self.input_layer(x)
        h = h + self.res_block(h)
        return self.output_layer(h).squeeze(-1)

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

def train_resnet_cv(
    embeddings_path,
    output_dir,
    epochs=100,
    batch_size=64,
    lr=1e-3,
    weight_decay=1e-4,
    hidden_dim=512,
    dropout=0.2,
    noise_level=0.0,
    patience=10,
    loss_type="mse",
    random_state=42,
    dry_run=False
):
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n==========================================")
    print(f"Starting 5-Fold CV ResNet MLP (Always Normalized)")
    print(f"Using Device: {device}")
    print(f"Saving checkpoints to: {output_dir}")
    print(f"==========================================")

    # Load embeddings and half_lives
    if dry_run:
        print("Running in DRY-RUN mode. Generating synthetic data...")
        embeddings = np.random.randn(100, 3072).astype(np.float32)
        half_lives = (np.abs(np.random.randn(100)) * 10).astype(np.float32)
    else:
        embeddings, _, half_lives = load_embeddings(embeddings_path)

    # Log transform targets (log1p)
    print("Applying log1p transformation to target half-lives...")
    half_lives = np.log1p(half_lives)

    # Bin the targets to stratify continuous target values
    print("Binning targets for stratification...")
    half_life_bins = pd.qcut(half_lives, q=10, labels=False, duplicates="drop")

    # Stratified K-Fold setup
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    fold_maes = []
    fold_mses = []
    fold_r2s = []
    all_loss_curves = []
    all_val_loss_curves = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(embeddings, half_life_bins)):
        fold_num = fold + 1
        print(f"\n--- Fold {fold_num}/5 ---")
        X_train, X_val = embeddings[train_idx], embeddings[val_idx]
        y_train, y_val = half_lives[train_idx], half_lives[val_idx]

        # Always normalize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # Convert to PyTorch datasets
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32), 
            torch.tensor(y_train, dtype=torch.float32)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32), 
            torch.tensor(y_val, dtype=torch.float32)
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Instantiate PyTorch model
        model = ResNetMLP(input_dim=embeddings.shape[1], hidden_dim=hidden_dim, dropout_rate=dropout, noise_level=noise_level).to(device)
        
        # Optimizer & Loss Function
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        if loss_type == "huber":
            criterion = nn.HuberLoss(delta=1.0)
        else:
            criterion = nn.MSELoss()

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)

        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        loss_curve = []
        val_loss_curve = []

        print(f"Training ResNet MLP on {len(X_train)} samples for {epochs} epochs...")
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * batch_x.size(0)
            
            epoch_loss /= len(X_train)
            loss_curve.append(epoch_loss)

            # Validation step to check for best checkpoint and update scheduler
            model.eval()
            with torch.no_grad():
                val_x_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
                val_y_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
                val_predictions = model(val_x_tensor)
                val_loss = criterion(val_predictions, val_y_tensor).item()
            
            val_loss_curve.append(val_loss)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == epochs - 1:
                print(f"  Epoch {epoch+1:03d}/{epochs:03d} | Train Loss: {epoch_loss:.5f} | Val Loss: {val_loss:.5f}")

            if patience_counter >= patience:
                print(f"  Early stopping triggered at epoch {epoch+1} (no improvement for {patience} epochs).")
                break

        # Load the best weights back
        model.load_state_dict(best_model_state)
        model.to(device)

        # Save the fold checkpoint
        checkpoint_path = os.path.join(output_dir, f"resnet_model_fold_{fold_num}.pkl")
        checkpoint_data = {
            "state_dict": best_model_state,
            "scaler": scaler,
            "loss_curve": loss_curve,
            "val_loss_curve": val_loss_curve,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "weight_decay": weight_decay,
            "noise_level": noise_level,
            "patience": patience,
            "input_dim": embeddings.shape[1]
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

        # Predict on validation set
        model.eval()
        with torch.no_grad():
            val_x_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
            y_pred = model(val_x_tensor).cpu().numpy()

        # Metrics (expm1 to evaluate in original scale)
        y_val_orig = np.expm1(y_val)
        y_pred_orig = np.expm1(y_pred)
        mse = mean_squared_error(y_val_orig, y_pred_orig)
        mae = mean_absolute_error(y_val_orig, y_pred_orig)
        r2 = r2_score(y_val_orig, y_pred_orig)

        print(f"Fold {fold_num} results: MAE = {mae:.4f}, MSE = {mse:.4f}, R^2 = {r2:.4f}")
        fold_maes.append(mae)
        fold_mses.append(mse)
        fold_r2s.append(r2)
        all_loss_curves.append(loss_curve)
        all_val_loss_curves.append(val_loss_curve)

    # Save aggregated loss curves as CSV
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

    # Calculate average scores
    avg_mae = np.mean(fold_maes)
    avg_mse = np.mean(fold_mses)
    avg_r2 = np.mean(fold_r2s)

    print(f"\nSummary (Always Normalized):")
    print(f"  Avg MAE: {avg_mae:.4f}")
    print(f"  Avg MSE: {avg_mse:.4f}")
    print(f"  Avg R^2: {avg_r2:.4f}")

    return {
        'mae': avg_mae,
        'mse': avg_mse,
        'r2': avg_r2
    }

def run_final_test_evaluation(
    test_path,
    best_config_dir,
    dry_run=False
):
    print(f"\n==========================================")
    print(f"Running Final Test Evaluation from: {best_config_dir}")
    print(f"==========================================")
    
    # Load test set
    if dry_run:
        print("Running in DRY-RUN mode. Generating synthetic test data...")
        X_test = np.random.randn(20, 3072).astype(np.float32)
        y_test = (np.abs(np.random.randn(20)) * 10).astype(np.float32)
    else:
        X_test, _, y_test = load_embeddings(test_path)

    # Log transform test targets (log1p)
    y_test = np.log1p(y_test)

    all_preds = []
    fold_mses = []
    fold_pearsons = []
    fold_spearmans = []

    for fold_idx in range(5):
        fold_num = fold_idx + 1
        model_path = os.path.join(best_config_dir, f"resnet_model_fold_{fold_num}.pkl")
        print(f"Loading checkpoint for fold {fold_num} from {model_path}...")
        
        with open(model_path, 'rb') as f:
            checkpoint = pickle.load(f)
            
        scaler = checkpoint["scaler"]
        state_dict = checkpoint["state_dict"]
        hidden_dim = checkpoint["hidden_dim"]
        dropout = checkpoint["dropout"]
        input_dim = checkpoint["input_dim"]
        noise_level = checkpoint.get("noise_level", 0.0)

        # Apply fold scaler if present
        if scaler is not None:
            X_test_fold = scaler.transform(X_test)
        else:
            X_test_fold = X_test

        # Reconstruct PyTorch Model
        model = ResNetMLP(input_dim=input_dim, hidden_dim=hidden_dim, dropout_rate=dropout, noise_level=noise_level)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        # Predict
        with torch.no_grad():
            x_test_tensor = torch.tensor(X_test_fold, dtype=torch.float32).to(device)
            y_pred_fold = model(x_test_tensor).cpu().numpy()
        
        all_preds.append(y_pred_fold)

        # Calculate fold-specific metrics in original scale (expm1)
        y_pred_fold_orig = np.expm1(y_pred_fold)
        y_test_orig = np.expm1(y_test)
        mse_f = np.mean((y_pred_fold_orig - y_test_orig) ** 2)
        df_f = pd.DataFrame({"prediction": y_pred_fold_orig, "label": y_test_orig})
        pearson_f = df_f.corr(method='pearson').iloc[0, 1]
        spearman_f = df_f.corr(method='spearman').iloc[0, 1]

        fold_mses.append(mse_f)
        fold_pearsons.append(pearson_f)
        fold_spearmans.append(spearman_f)

    # Average predictions of the 5 models (ensemble prediction)
    y_pred_ensemble = np.mean(all_preds, axis=0)

    # Calculate statistics across folds
    mean_mse, std_mse = np.mean(fold_mses), np.std(fold_mses)
    mean_pearson, std_pearson = np.mean(fold_pearsons), np.std(fold_pearsons)
    mean_spearman, std_spearman = np.mean(fold_spearmans), np.std(fold_spearmans)

    # Generate results report text
    res_text = f"----- ResNet MLP Evaluation -----\n"
    for i in range(5):
        res_text += f"Fold {i+1}:\n"
        res_text += f"  MSE: {fold_mses[i]}\n"
        res_text += f"  Pearson: {fold_pearsons[i]}\n"
        res_text += f"  Spearman: {fold_spearmans[i]}\n"
    
    res_text += f"\nMean +- std (across 5 folds):\n"
    res_text += f"  MSE: {mean_mse} +- {std_mse}\n"
    res_text += f"  Pearson: {mean_pearson} +- {std_pearson}\n"
    res_text += f"  Spearman: {mean_spearman} +- {std_spearman}\n"

    print(res_text)

    # Save report to resnet_results.txt
    results_path = os.path.join(best_config_dir, "resnet_results.txt")
    with open(results_path, "w") as f:
        f.write(res_text)
    print(f"Saved evaluation results to: {results_path}")

    # Save test predictions CSV (ensemble predictions in original scale)
    y_test_orig = np.expm1(y_test)
    y_pred_ensemble_orig = np.expm1(y_pred_ensemble)
    df_out = pd.DataFrame({
        "label": y_test_orig,
        "prediction": y_pred_ensemble_orig
    })
    out_csv_path = os.path.join(best_config_dir, "test_predictions.csv")
    df_out.to_csv(out_csv_path, index=False)
    print(f"Saved ensemble test predictions to: {out_csv_path}")

def run_training_experiment(
    train_val_path,
    test_path,
    output_dir,
    epochs=100,
    batch_size=64,
    lr=1e-3,
    weight_decay=1e-4,
    hidden_dim=512,
    dropout=0.2,
    noise_level=0.0,
    patience=10,
    loss_type="mse",
    dry_run=False
):
    # Run CV (saves fold checkpoints directly under output_dir)
    train_resnet_cv(
        embeddings_path=train_val_path,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        hidden_dim=hidden_dim,
        dropout=dropout,
        noise_level=noise_level,
        patience=patience,
        loss_type=loss_type,
        dry_run=dry_run
    )
    
    # Run final test set evaluation using the loaded ensemble models
    run_final_test_evaluation(
        test_path=test_path,
        best_config_dir=output_dir,
        dry_run=dry_run
    )

def main():
    parser = argparse.ArgumentParser(description="Train ResNet MLP in PyTorch on AlphaGenome embeddings.")
    parser.add_argument(
        "--split",
        action="store_true",
        help="Split original embeddings.npz into train_val and test sets."
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the ResNet MLP on the train_val embeddings using 5-fold CV and evaluate on the test set."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run final test evaluation on the test set using already trained models."
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
        default="/beegfs/prj/RNA_NLP/AlphaGenome/resnet_results",
        help="Directory to save the final models, loss CSVs and plot PNGs."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs to train the model."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for AdamW optimizer."
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay (L2 penalty) for AdamW optimizer."
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=512,
        help="Hidden dimension size of ResNetMLP layers."
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout probability."
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=["mse", "huber"],
        default="mse",
        help="Loss function type to use for training."
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Patience (epochs) for early stopping."
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="Standard deviation of Gaussian noise added to inputs during training."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a dry run with synthetic data to test the pipeline end-to-end."
    )
    args = parser.parse_args()

    if args.split:
        split_embeddings()
    elif args.train or args.dry_run:
        # For dry-run, set epochs to a small number
        epochs = 2 if args.dry_run else args.epochs
        run_training_experiment(
            train_val_path=args.embeddings_path,
            test_path=args.test_path,
            output_dir=args.output_dir,
            epochs=epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            noise_level=args.noise,
            patience=args.patience,
            loss_type=args.loss,
            dry_run=args.dry_run
        )
    elif args.test:
        # Check if checkpoints exist directly in the output directory
        has_checkpoints = os.path.exists(args.output_dir) and any(f.endswith(".pkl") for f in os.listdir(args.output_dir))
        
        if has_checkpoints:
            best_config_dir = args.output_dir
        else:
            # Fallback check for older "with_normalization" subfolder structure
            with_norm_dir = os.path.join(args.output_dir, "with_normalization")
            if os.path.exists(with_norm_dir) and any(f.endswith(".pkl") for f in os.listdir(with_norm_dir)):
                best_config_dir = with_norm_dir
            else:
                print(f"Error: No trained models/checkpoints found in {args.output_dir}")
                return
            
        run_final_test_evaluation(
            test_path=args.test_path,
            best_config_dir=best_config_dir
        )
    else:
        # If no explicit flag is set, run standard training if file exists
        if os.path.exists(args.embeddings_path):
            print(f"Found {args.embeddings_path}. Running training experiment...")
            run_training_experiment(
                train_val_path=args.embeddings_path,
                test_path=args.test_path,
                output_dir=args.output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                loss_type=args.loss
            )
        else:
            parser.print_help()

if __name__ == "__main__":
    main()
