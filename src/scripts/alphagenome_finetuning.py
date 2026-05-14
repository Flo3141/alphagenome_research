import time
import math
import os
import jax
import optax
import orbax.checkpoint as ocp
import numpy as np
from alphagenome.models import dna_model
from alphagenome_research.finetuning import finetune
from alphagenome_research.model.metadata import metadata as metadata_lib

# --- Die Speicherfunktion von vorhin ---
def save_head_checkpoint(params, state, checkpoint_dir, step_name, checkpointer):
    """Speichert ausschließlich die Gewichte und Zustände des RNA-Heads."""
    import shutil
    path = os.path.abspath(os.path.join(checkpoint_dir, str(step_name)))
    if os.path.exists(path):
        shutil.rmtree(path)
    head_params = {k: v for k, v in params.items() if 'rna_half_life' in k}
    head_state = {k: v for k, v in state.items() if 'rna_half_life' in k}
    checkpointer.save(path, (head_params, head_state))
    print(f"--> Checkpoint saved at: {path}")

if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # 1. SETUP & PFADE
    # -------------------------------------------------------------------------
    data_folder = os.environ.get("AG_DATA_FOLDER", ".")
    train_csv_path = os.path.join(data_folder, "half_life_with_coords_train.csv") 
    val_csv_path = os.path.join(data_folder, "half_life_with_coords_val.csv")     
    test_csv_path = os.path.join(data_folder, "half_life_with_coords_test.csv")
    
    alphagenome_checkpoint_path = "/beegfs/prj/RNA_NLP/AlphaGenome/weights/alphagenome/all_folds/1"
    checkpoint_dir = "/beegfs/prj/RNA_NLP/AlphaGenome/weights/checkpoints_rna_half_life"
    
    batch_size = 4
    sequence_length = 524_288 // 8
    num_epochs = 2  # Wie oft über den gesamten Datensatz iteriert wird
    
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
        (loss, scalars, _), _ = forward_fn.apply(
            params, state, rng, batch, is_training=False # Wichtig: Schaltet Dropout aus!
        )
        scalars['loss'] = loss
        return scalars

    # -------------------------------------------------------------------------
    # 4. TRAINING INITIALISIEREN (Dummy-Batch holen)
    # -------------------------------------------------------------------------
    print("Initializing training...")
    dummy_iter = finetune.get_rna_half_life_dataset_iterator(
        batch_size=batch_size, sequence_length=sequence_length, labels_csv_path=train_csv_path
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
    # 5. DIE EPOCHEN-LOOP (TRAIN & VALIDATION)
    # -------------------------------------------------------------------------
    print("Starting training loop...")
    rng = jax.random.PRNGKey(42)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\n========== EPOCH {epoch+1}/{num_epochs} ==========")
        with open(train_csv_path, 'r', encoding='utf-8') as f:
            num_samples = sum(1 for line in f) - 1  # -1 to exclude the header row

        num_train_batches = math.ceil(num_samples / batch_size)

        # --- A. TRAINING PHASE ---
        # Wir erstellen den Iterator für jede Epoche neu, damit er von vorne beginnt
        train_iter = finetune.get_rna_half_life_dataset_iterator(
            batch_size=batch_size, sequence_length=sequence_length, 
            labels_csv_path=train_csv_path, shuffle=True
        )
        
        train_loss_sum = 0.0
        train_mae_sum = 0.0
        num_train_steps = 0
        
        print(f"Training with {num_train_batches} batches")
        start_time = time.time()
        for step, batch in enumerate(train_iter):
            rng, step_rng = jax.random.split(rng)
            params, state, opt_state, scalars = train_step(params, state, opt_state, step_rng, batch)
            
            train_loss_sum += scalars['loss'].item()
            train_mae_sum += scalars['rna_half_life_mae'].item()
            num_train_steps += 1

            if step == 2:
                print(f"  Time elapsed after 2 steps: {time.time() - start_time:.2f}s")
                print(f"  Estimated time for 500 steps: {(time.time() - start_time)/2*500}s")
                print(f"  Estimated time for full epoch: {(time.time() - start_time)/2*num_train_batches}s")
                print(f"  Estimated time for full training: {(time.time() - start_time)/2*num_train_batches*num_epochs}s")
                print(f"  Estimated time for full training in minutes: {((time.time() - start_time)/2*num_train_batches*num_epochs)/60}min")
                print(f"  Estimated time for full training in hours: {((time.time() - start_time)/2*num_train_batches*num_epochs)/3600}h")
                print(f"  Estimated time for full training in days: {((time.time() - start_time)/2*num_train_batches*num_epochs)/86400}days")
            
            # Alle 500 Batches ein kurzes Lebenszeichen drucken
            if step > 0 and step % 500 == 0:
                elapsed = time.time() - start_time
                steps_done = step + 1
                steps_remaining = max(0, num_train_batches - steps_done)
                eta_min = (elapsed / steps_done) * steps_remaining / 60
                print(f"  Step {step} | Current Loss: {scalars['loss'].item():.4f}")
                print(f"  Time elapsed: {elapsed:.2f}s --> {eta_min:.2f} minutes remaining (estimated {num_train_batches} batches total)")


        avg_train_loss = train_loss_sum / max(1, num_train_steps)
        avg_train_mae = train_mae_sum / max(1, num_train_steps)


        # --- B. VALIDATION PHASE ---
        # Wichtig: shuffle=False bei Validierung!
        val_iter = finetune.get_rna_half_life_dataset_iterator(
            batch_size=batch_size, sequence_length=sequence_length, 
            labels_csv_path=val_csv_path, shuffle=False
        )
        
        val_loss_sum = 0.0
        val_mae_sum = 0.0
        num_val_steps = 0
        
        print(f"Validating...")
        for batch in val_iter:
            rng, eval_rng = jax.random.split(rng)
            val_scalars = eval_step(params, state, eval_rng, batch)
            
            val_loss_sum += val_scalars['loss'].item()
            val_mae_sum += val_scalars['rna_half_life_mae'].item()
            num_val_steps += 1

        avg_val_loss = val_loss_sum / max(1, num_val_steps)
        avg_val_mae = val_mae_sum / max(1, num_val_steps)

        # --- C. ERGEBNISSE & CHECKPOINTING ---
        print(f"-> Train Loss: {avg_train_loss:.4f} | Train MAE: {avg_train_mae:.4f}")
        print(f"-> Val Loss:   {avg_val_loss:.4f} | Val MAE:   {avg_val_mae:.4f}")

        # Speichere das Modell NUR, wenn es auf dem Val-Set besser geworden ist (Early Stopping Logik)
        if avg_val_loss < best_val_loss:
            print(f"🌟 Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}!")
            best_val_loss = avg_val_loss
            save_head_checkpoint(params, state, checkpoint_dir, "best_model", checkpointer)
            
        # Optional: Jede Epoche speichern, falls du den Verlauf analysieren willst
        save_head_checkpoint(params, state, checkpoint_dir, f"epoch_{epoch+1}", checkpointer)

    print("\nTraining finished! Best model is saved as 'best_model'.")

    # -------------------------------------------------------------------------
    # 6. FINALE EVALUIERUNG AUF DEM TEST-SET
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

    print("Running inference on test dataset...")
    # D. Über das gesamte Test-Set iterieren
    for batch in test_iter:
        rng, eval_rng = jax.random.split(rng)
        
        # Wir nutzen wieder eval_step (is_training=False), damit Dropout aus ist!
        test_scalars = eval_step(params, state, eval_rng, batch)
        
        test_loss_sum += test_scalars['loss'].item()
        test_mae_sum += test_scalars['rna_half_life_mae'].item()
        num_test_steps += 1

    # E. Durchschnittliche Fehler berechnen
    avg_test_loss = test_loss_sum / max(1, num_test_steps)
    avg_test_mae = test_mae_sum / max(1, num_test_steps)

    print("\n" + "="*50)
    print(" FINAL TEST RESULTS")
    print("="*50)
    print(f" Test Loss (Huber): {avg_test_loss:.4f}")
    print(f" Test MAE (log1p):  {avg_test_mae:.4f}")
    print("="*50)