import jax
import optax
import orbax.checkpoint as ocp  # <--- Hinzugefügt
from alphagenome.models import dna_model
from alphagenome_research.finetuning import finetune
from alphagenome_research.model.metadata import metadata as metadata_lib
import os

def save_head_checkpoint(params, state, checkpoint_dir, step_name, checkpointer):
    """Speichert ausschließlich die Gewichte und Zustände des RNA-Heads."""
    
    # Pfad erstellen (absolut ist sicherer auf Clustern)
    path = os.path.abspath(os.path.join(checkpoint_dir, str(step_name)))
    
    # Filtern: Nur Einträge behalten, die 'rna_half_life' im Namen haben
    head_params = {k: v for k, v in params.items() if 'rna_half_life' in k}
    head_state = {k: v for k, v in state.items() if 'rna_half_life' in k}
    
    # Speichern via Orbax
    checkpointer.save(
        path, 
        args=ocp.args.StandardSave((head_params, head_state))
    )
    print(f"--> Checkpoint (Head-only) saved at: {path}")

if __name__ == "__main__":
    data_folder = os.environ.get("AG_DATA_FOLDER", ".")
    input_csv_path = os.path.join(data_folder, "half_life_with_coords.csv")
    # 1. Metadaten laden
    print("Loading metadata...")
    # output_metadata = {
    #     dna_model.Organism.HOMO_SAPIENS: metadata_lib.load(
    #         dna_model.Organism.HOMO_SAPIENS
    #     )
    # }
    output_metadata = {}
    # 2. Vorrainierte Gewichte aus Checkpoint laden  <--- NEU
    print("Loading pretrained weights...")
    alphagenome_checkpoint_path = "/beegfs/prj/RNA_NLP/AlphaGenome/weights/alphagenome/all_folds/1"
    checkpointer = ocp.StandardCheckpointer()
    # Der Checkpoint gibt ein Tuple (params, state) zurück
    pretrained_params, pretrained_state = checkpointer.restore(alphagenome_checkpoint_path)
    checkpoint_dir = "/beegfs/prj/RNA_NLP/AlphaGenome/weights/checkpoints_rna_half_life"
    # 3. Forward-Funktion erstellen
    print("Creating forward function...")
    forward_fn = finetune.get_rna_half_life_forward_fn(output_metadata)
    # 4. Dataset-Iterator erstellen
    print("Creating dataset iterator...")
    data_iter = finetune.get_rna_half_life_dataset_iterator(
        batch_size=4,
        sequence_length=524_288 // 8,
        labels_csv_path=input_csv_path,
    )
    # 5. Training initialisieren (jetzt mit geladenen Checkpoint-Daten)
    print("Initializing training...")
    params, state, opt_state, train_step = finetune.init_rna_half_life_training(
        forward_fn=forward_fn,
        base_optimizer=optax.adam(1e-4),
        sample_batch=next(data_iter),
        pretrained_params=pretrained_params,  # <--- Hier übergeben
        pretrained_state=pretrained_state,    # <--- Hier übergeben
    )
    # 6. Trainings-Loop
    print("Starting training loop...")
    rng = jax.random.PRNGKey(42)
    for step, batch in enumerate(data_iter):
        rng, step_rng = jax.random.split(rng)
        params, state, opt_state, scalars = train_step(params, state, opt_state, rng, batch)
        print(f"Loss: {scalars['loss'].item():.4f}, MAE: {scalars['rna_half_life_mae'].item():.4f}")
        # 2. Regelmäßig speichern (z.B. alle 100 Schritte)
        if step > 0 and step % 100 == 0:
            save_head_checkpoint(params, state, checkpoint_dir, f"step_{step}", checkpointer)
    # 3. Ganz am Ende des Trainings den finalen Checkpoint speichern
    save_head_checkpoint(params, state, checkpoint_dir, "final_model", checkpointer)
    print(f"--> Training finished! Final model saved at {checkpoint_dir}/final_model")