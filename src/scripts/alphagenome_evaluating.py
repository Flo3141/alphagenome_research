import time
import math
import os
import jax
import optax
import orbax.checkpoint as ocp
import numpy as np
import pandas as pd
from alphagenome.models import dna_model
from alphagenome_research.finetuning import finetune
from alphagenome_research.model.metadata import metadata as metadata_lib

def evaluate_predictions(path):
    res_text = ""
    test_pred = os.path.join(path, f"test_predictions.csv")
    df = pd.read_csv(test_pred)
    df.drop(columns=['sequence'], inplace=True)
    predict_labels = df["prediction"]
    true_labels = df["label"]
    all_mse = np.mean((np.array(predict_labels) - np.array(true_labels)) ** 2)
    all_prearson = df.corr(method='pearson').iloc[0, 1]
    all_spearman = df.corr(method='spearman').iloc[0, 1]
    res_text = f"----- AlphaGenome Evaluation -----\n"
    res_text += f"MSE: {all_mse}\n"
    res_text += f"Pearson: {all_prearson}\n"
    res_text += f"Spearman: {all_spearman}\n"
    
    print(res_text)
    with open(os.path.join(path, "alphagenome_results.txt"), "w") as f:
        f.write(res_text)

def create_predictions(path):
    # -------------------------------------------------------------------------
    # 1. SETUP & PFADE
    # -------------------------------------------------------------------------
    data_folder = os.environ.get("AG_DATA_FOLDER", ".")
    test_csv_path = os.path.join(data_folder, "half_life_with_coords_test.csv")
    
    alphagenome_checkpoint_path = "/beegfs/prj/RNA_NLP/AlphaGenome/weights/alphagenome/all_folds/1"
    checkpoint_dir = "/beegfs/prj/RNA_NLP/AlphaGenome/weights/checkpoints_rna_half_life"
    
    batch_size = 4
    sequence_length = 524_288 // 8
    
    # -------------------------------------------------------------------------
    # 2. MODELL & GEWICHTE LADEN
    # -------------------------------------------------------------------------
    print("Loading pretrained weights...")
    checkpointer = ocp.StandardCheckpointer()
    pretrained_params, pretrained_state = checkpointer.restore(alphagenome_checkpoint_path)

    print("Creating forward function...")
    # Leere Metadaten, da wir nur den Half-Life Head brauchen
    forward_fn = finetune.get_rna_half_life_forward_fn({})

    # -------------------------------------------------------------------------
    # 3. EVALUATION STEP DEFINIEREN
    # -------------------------------------------------------------------------
    @jax.jit
    def eval_step(params, state, rng, batch):
        """Berechnet den Loss, aber aktualisiert KEINE Gewichte (is_training=False)."""
        (loss, scalars, preds), _ = forward_fn.apply(
            params, state, rng, batch, is_training=False # Wichtig: Schaltet Dropout aus!
        )
        scalars['loss'] = loss
        return scalars, preds['rna_half_life']['predictions']

    # -------------------------------------------------------------------------
    # 4. TRAINING INITIALISIEREN (Dummy-Batch holen)
    # -------------------------------------------------------------------------
    print("Initializing training...")
    dummy_iter = finetune.get_rna_half_life_dataset_iterator(
        batch_size=batch_size, sequence_length=sequence_length, labels_csv_path=test_csv_path
    )
    params, state, opt_state, train_step = finetune.init_rna_half_life_training(
        forward_fn=forward_fn,
        base_optimizer=optax.adam(1e-4),
        sample_batch=next(dummy_iter),
        pretrained_params=pretrained_params,
        pretrained_state=pretrained_state,
    )
    del dummy_iter # Brauchen wir nicht mehr

    # -------------------------------------------------------------------------
    # 5. FINALE EVALUIERUNG AUF DEM TEST-SET
    # -------------------------------------------------------------------------
    print("\n\n" + "="*50)
    print(" STARTING FINAL TEST EVALUATION")
    print("="*50)

    # B. Das "beste" Modell laden
    best_model_path = os.path.abspath(os.path.join(checkpoint_dir, "best_model"))
    print(f"Loading best validation model from: {best_model_path}")
    
    # Checkpoint laden (enthält nur die 'rna_half_life' Gewichte)
    best_head_params, best_head_state = checkpointer.restore(best_model_path)
    
    # Da params und state Dictionaries sind, können wir die geladenen 
    # besten Gewichte einfach über die aktuellen Gewichte drüberbügeln (mergen)
    params.update(best_head_params)
    state.update(best_head_state)

    # C. Test-Iterator erstellen (Wichtig: shuffle=False!)
    test_iter = finetune.get_rna_half_life_dataset_iterator(
        batch_size=batch_size, 
        sequence_length=sequence_length, 
        labels_csv_path=test_csv_path, 
        shuffle=False
    )

    test_loss_sum = 0.0
    test_mae_sum = 0.0
    num_test_steps = 0
    all_preds = []

    print("Running inference on test dataset...")
    # D. Über das gesamte Test-Set iterieren
    for batch in test_iter:
        rng, eval_rng = jax.random.split(rng)
        
        # Wir nutzen wieder eval_step (is_training=False), damit Dropout aus ist!
        test_scalars, batch_preds = eval_step(params, state, eval_rng, batch)
        
        test_loss_sum += test_scalars['loss'].item()
        test_mae_sum += test_scalars['rna_half_life_mae'].item()
        num_test_steps += 1
        all_preds.append(np.array(batch_preds))

    # E. Durchschnittliche Fehler berechnen
    avg_test_loss = test_loss_sum / max(1, num_test_steps)
    avg_test_mae = test_mae_sum / max(1, num_test_steps)

    print("\n" + "="*50)
    print(" FINAL TEST RESULTS")
    print("="*50)
    print(f" Test Loss (Huber): {avg_test_loss:.4f}")
    print(f" Test MAE (log1p):  {avg_test_mae:.4f}")
    print("="*50)

    # F. CSV exportieren
    all_preds_flat = np.concatenate(all_preds, axis=0)
    
    # CSV laden um Sequenz-IDs und echte Labels zu bekommen
    df_test = pd.read_csv(test_csv_path)
    if len(all_preds_flat) < len(df_test):
        print(f"Warning: Number of predictions ({len(all_preds_flat)}) < Number of rows in CSV ({len(df_test)}). Truncating df_test.")
        df_test = df_test.iloc[:len(all_preds_flat)]
        
    df_out = pd.DataFrame()
    # Verschiedene Spaltennamen für die Sequenz-ID ausprobieren
    if 'sequence' in df_test.columns:
        df_out['sequence'] = df_test['sequence']
    elif 'id' in df_test.columns:
        df_out['sequence'] = df_test['id']
    elif 'transcript_id' in df_test.columns:
        df_out['sequence'] = df_test['transcript_id']
    else:
        df_out['sequence'] = df_test.index
        
    df_out['label'] = df_test['half_life']
    df_out['prediction'] = all_preds_flat
    
    out_csv_path = os.path.join(checkpoint_dir, "test_predictions.csv")
    df_out.to_csv(out_csv_path, index=False)
    print(f"Saved predictions to {out_csv_path}")

if __name__ == "__main__":
    create_predictions(".")
    evaluate_predictions(".")
