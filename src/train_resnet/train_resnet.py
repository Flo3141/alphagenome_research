import argparse
import os
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ResNetMLP import ResNetMLP
from util import load_embeddings, split_embeddings, save_loss_curves, save_loss_curves_as_csv
from bayes_opt import BayesianOptimization

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_step(model, criterion, optimizer, train_loader, device):
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
    
    return epoch_loss / len(train_loader.dataset)


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
    dry_run=False,
    save_checkpoints=True
):
    if save_checkpoints:
        os.makedirs(output_dir, exist_ok=True)

    print(f"\n==========================================")
    print(f"Starting 5-Fold CV ResNet MLP (Always Normalized)")
    print(f"Using Device: {device}")
    if save_checkpoints:
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
    fold_pearsons = []
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
            epoch_loss = train_step(model, criterion, optimizer, train_loader, device)
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

        if save_checkpoints:
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
            save_loss_curves(loss_curve, val_loss_curve, fold_num, output_dir, loss_type)

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

        # Calculate validation Pearson Correlation Coefficient
        df_val = pd.DataFrame({"prediction": y_pred_orig.flatten(), "label": y_val_orig.flatten()})
        pearson_val = df_val.corr(method='pearson').iloc[0, 1]
        if np.isnan(pearson_val):
            pearson_val = -1.0

        print(f"Fold {fold_num} results: MAE = {mae:.4f}, MSE = {mse:.4f}, R^2 = {r2:.4f}, Pearson = {pearson_val:.4f}")
        fold_maes.append(mae)
        fold_mses.append(mse)
        fold_r2s.append(r2)
        fold_pearsons.append(pearson_val)
        all_loss_curves.append(loss_curve)
        all_val_loss_curves.append(val_loss_curve)

    # Save aggregated loss curves as CSV
    if save_checkpoints:
        save_loss_curves_as_csv(all_loss_curves, all_val_loss_curves, output_dir, loss_type)

    # Calculate average scores
    avg_mae = np.mean(fold_maes)
    avg_mse = np.mean(fold_mses)
    avg_r2 = np.mean(fold_r2s)
    avg_pearson = np.mean(fold_pearsons)

    print(f"\nSummary (Always Normalized):")
    print(f"  Avg MAE: {avg_mae:.4f}")
    print(f"  Avg MSE: {avg_mse:.4f}")
    print(f"  Avg R^2: {avg_r2:.4f}")
    print(f"  Avg Pearson: {avg_pearson:.4f}")

    return {
        'mae': avg_mae,
        'mse': avg_mse,
        'r2': avg_r2,
        'pearson': avg_pearson
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

def run_bayesian_optimization(
    embeddings_path,
    output_dir,
    epochs,
    batch_size,
    patience,
    loss_type,
    dry_run,
    lr_args,
    weight_decay_args,
    hidden_dim_args,
    dropout_args,
    noise_args,
    init_points=5,
    n_iter=15,
    random_state=42
):
    print("\nInitializing Bayesian Optimization...")
    params_dict = {
        "lr": lr_args,
        "weight_decay": weight_decay_args,
        "hidden_dim": hidden_dim_args,
        "dropout": dropout_args,
        "noise": noise_args
    }
    
    for name, val in params_dict.items():
        if len(val) not in [1, 2]:
            raise ValueError(f"Argument --{name} must have either 1 value (constant) or 2 values (range [min, max]). Got: {val}")

    pbounds = {}
    constants = {}
    for name, val in params_dict.items():
        if len(val) == 2:
            pbounds[name] = (val[0], val[1])
        else:
            constants[name] = val[0]

    print(f"Optimizing parameters over ranges: {pbounds}")
    print(f"Keeping parameters constant: {constants}")

    def objective_func(**kwargs):
        eval_params = {}
        for name in params_dict.keys():
            if name in kwargs:
                if name == "hidden_dim":
                    eval_params[name] = int(round(kwargs[name]))
                else:
                    eval_params[name] = kwargs[name]
            else:
                if name == "hidden_dim":
                    eval_params[name] = int(round(constants[name]))
                else:
                    eval_params[name] = constants[name]

        print(f"\nEvaluating with parameters: {eval_params}")
        
        cv_results = train_resnet_cv(
            embeddings_path=embeddings_path,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            lr=eval_params["lr"],
            weight_decay=eval_params["weight_decay"],
            hidden_dim=eval_params["hidden_dim"],
            dropout=eval_params["dropout"],
            noise_level=eval_params["noise"],
            patience=patience,
            loss_type=loss_type,
            random_state=random_state,
            dry_run=dry_run,
            save_checkpoints=False
        )
        return cv_results["pearson"]

    optimizer = BayesianOptimization(
        f=objective_func,
        pbounds=pbounds,
        random_state=random_state,
        verbose=2
    )

    optimizer.maximize(
        init_points=init_points,
        n_iter=n_iter
    )

    best_targets = optimizer.max["target"]
    best_params = optimizer.max["params"]
    print(f"\nBayesian Optimization completed! Best validation Pearson Correlation: {best_targets:.4f}")
    print(f"Best parameters: {best_params}")

    final_params = {}
    for name, val in params_dict.items():
        if name in best_params:
            if name == "hidden_dim":
                final_params[name] = int(round(best_params[name]))
            else:
                final_params[name] = float(best_params[name])
        else:
            if name == "hidden_dim":
                final_params[name] = int(round(val[0]))
            else:
                final_params[name] = float(val[0])

    return final_params

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
        nargs="+",
        type=float,
        default=[1e-3],
        help="Learning rate for AdamW optimizer (single float or range [min, max] for Bayesian optimization)."
    )
    parser.add_argument(
        "--weight_decay",
        nargs="+",
        type=float,
        default=[1e-4],
        help="Weight decay (L2 penalty) for AdamW optimizer (single float or range [min, max] for Bayesian optimization)."
    )
    parser.add_argument(
        "--hidden_dim",
        nargs="+",
        type=float,
        default=[512.0],
        help="Hidden dimension size of ResNetMLP layers (single float or range [min, max] for Bayesian optimization)."
    )
    parser.add_argument(
        "--dropout",
        nargs="+",
        type=float,
        default=[0.2],
        help="Dropout probability (single float or range [min, max] for Bayesian optimization)."
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
        nargs="+",
        type=float,
        default=[0.0],
        help="Percentage of the standard deviation of Gaussian noise added to inputs during training (0.02 --> 2% of the std of the data) (single float or range [min, max] for Bayesian optimization)."
    )
    parser.add_argument(
        "--bayes_init_points",
        type=int,
        default=5,
        help="Number of initial random exploration points for Bayesian optimization."
    )
    parser.add_argument(
        "--bayes_n_iter",
        type=int,
        default=15,
        help="Number of iterations for Bayesian optimization search."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a dry run with synthetic data to test the pipeline end-to-end."
    )
    args = parser.parse_args()

    # Validate argument lengths
    for name, val in [("lr", args.lr), ("weight_decay", args.weight_decay), ("hidden_dim", args.hidden_dim), ("dropout", args.dropout), ("noise", args.noise)]:
        if len(val) not in [1, 2]:
            raise ValueError(f"Argument --{name} must have either 1 value (constant) or 2 values (range [min, max]). Got: {val}")

    is_bayes_opt = any(len(val) == 2 for val in [args.lr, args.weight_decay, args.hidden_dim, args.dropout, args.noise])

    if args.split:
        split_embeddings()
    elif args.train or args.dry_run or (not args.test and os.path.exists(args.embeddings_path)):
        if not args.train and not args.dry_run and os.path.exists(args.embeddings_path):
            print(f"Found {args.embeddings_path}. Running training experiment...")

        # For dry-run, set epochs to a small number
        epochs = 2 if args.dry_run else args.epochs

        if is_bayes_opt:
            print("\n==========================================")
            print("Running Bayesian Optimization for Hyperparameters")
            print("==========================================")
            best_params = run_bayesian_optimization(
                embeddings_path=args.embeddings_path,
                output_dir=args.output_dir,
                epochs=epochs,
                batch_size=args.batch_size,
                patience=args.patience,
                loss_type=args.loss,
                dry_run=args.dry_run,
                lr_args=args.lr,
                weight_decay_args=args.weight_decay,
                hidden_dim_args=args.hidden_dim,
                dropout_args=args.dropout,
                noise_args=args.noise,
                init_points=args.bayes_init_points,
                n_iter=args.bayes_n_iter,
                random_state=42
            )

            # Format folder name with best parameters and create it
            folder_name = (
                f"best_lr{best_params['lr']:.2e}_"
                f"wd{best_params['weight_decay']:.2e}_"
                f"hd{best_params['hidden_dim']}_"
                f"dr{best_params['dropout']:.2f}_"
                f"ns{best_params['noise']:.2f}"
            ).replace("+", "")

            best_config_dir = os.path.join(args.output_dir, folder_name)
            os.makedirs(best_config_dir, exist_ok=True)

            # Save the config file
            config_path = os.path.join(best_config_dir, "config.json")
            with open(config_path, "w") as f:
                json.dump(best_params, f, indent=4)
            print(f"\nSaved optimal configuration to: {config_path}")

            print("\n==========================================")
            print("Running final training with optimal hyperparameters")
            print(f"Output directory: {best_config_dir}")
            print("==========================================")

            run_training_experiment(
                train_val_path=args.embeddings_path,
                test_path=args.test_path,
                output_dir=best_config_dir,
                epochs=epochs,
                batch_size=args.batch_size,
                lr=best_params["lr"],
                weight_decay=best_params["weight_decay"],
                hidden_dim=best_params["hidden_dim"],
                dropout=best_params["dropout"],
                noise_level=best_params["noise"],
                patience=args.patience,
                loss_type=args.loss,
                dry_run=args.dry_run
            )
        else:
            # Standard single-value run
            lr = args.lr[0]
            weight_decay = args.weight_decay[0]
            hidden_dim = int(round(args.hidden_dim[0]))
            dropout = args.dropout[0]
            noise_level = args.noise[0]

            run_training_experiment(
                train_val_path=args.embeddings_path,
                test_path=args.test_path,
                output_dir=args.output_dir,
                epochs=epochs,
                batch_size=args.batch_size,
                lr=lr,
                weight_decay=weight_decay,
                hidden_dim=hidden_dim,
                dropout=dropout,
                noise_level=noise_level,
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
        parser.print_help()

if __name__ == "__main__":
    main()
