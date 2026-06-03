import argparse
import os
import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import jmp
import orbax.checkpoint as ocp
import pandas as pd
from tqdm import tqdm

from alphagenome.data import genome
from alphagenome_research.io import fasta
from alphagenome_research.model import one_hot_encoder
from alphagenome_research.model import schemas
from alphagenome_research.model import model

def get_embeddings_forward_fn(output_metadata, jmp_policy='params=float32,compute=bfloat16,output=bfloat16'):
    """Erstellt die Forward-Funktion, die nur die Embeddings zurückgibt."""
    jmp_policy_obj = jmp.get_policy(jmp_policy)

    @hk.transform_with_state
    def forward(batch, is_training=False):
        with hk.mixed_precision.push_policy(model.AlphaGenome, jmp_policy_obj):
            alpha_genome_model = model.AlphaGenome(
                output_metadata, freeze_trunk_embeddings=True
            )
            
            # AlphaGenome.__call__ liefert (predictions, embeddings)
            predictions, embeddings = alpha_genome_model(
                batch.dna_sequence, batch.get_organism_index(), is_training=is_training
            )
            # Die 128bp Embeddings abrufen und wie im RNAHalfLifeHead über die 
            # Sequenzachse (axis=1) mitteln
            x = embeddings.get_sequence_embeddings(1)
            x_pooled = jnp.mean(x.astype(jnp.float32), axis=1)
            
            return x_pooled

    return forward

def main():
    data_folder = os.environ.get("AG_DATA_FOLDER", ".")
    default_csv = os.path.join(data_folder, "half_life_with_coords_train_12.csv")
    
    parser = argparse.ArgumentParser(description="Extrahiert AlphaGenome Embeddings für RNA Half-Life")
    parser.add_argument("--csv_path", type=str, default=default_csv, help="Pfad zur Eingabe CSV-Datei")
    parser.add_argument("--output_path", type=str, default="/beegfs/prj/RNA_NLP/AlphaGenome/embeddings/embeddings.npz", help="Speicherpfad für die Embeddings (.npy oder .npz)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch Size für die Inferenz")
    parser.add_argument("--checkpoint_path", type=str, default="/beegfs/prj/RNA_NLP/AlphaGenome/weights/alphagenome/all_folds/1", help="Pfad zum AlphaGenome Checkpoint")
    parser.add_argument("--fasta_path", type=str, default="https://storage.googleapis.com/alphagenome/reference/gencode/hg38/GRCh38.p13.genome.fa", help="Pfad oder URL zur Referenzgenom FASTA-Datei")
    args = parser.parse_args()

    sequence_length = 524_288 // 8

    print(f"Lade vortrainierte Gewichte von {args.checkpoint_path}...")
    checkpointer = ocp.StandardCheckpointer()
    pretrained_params, pretrained_state = checkpointer.restore(args.checkpoint_path)

    print("Erstelle Forward-Funktion...")
    forward_fn = get_embeddings_forward_fn({})

    @jax.jit
    def embed_step(params, state, batch):
        # Zufallsgenerator-Key (RNG) wird für is_training=False (kein Dropout) ignoriert, 
        # muss aber für die hk.transform_with_state Signatur übergeben werden.
        rng = jax.random.PRNGKey(0) 
        embeddings, _ = forward_fn.apply(params, state, rng, batch, is_training=False)
        return embeddings

    print(f"Initialisiere FastaExtractor und lade CSV von {args.csv_path}...")
    fasta_extractor = fasta.FastaExtractor(args.fasta_path)
    one_hot_enc = one_hot_encoder.DNAOneHotEncoder()
    
    df = pd.read_csv(args.csv_path)
    num_samples = len(df)
    print(f"Gefundene Sequenzen in CSV: {num_samples}")

    def get_batches():
        batch_seqs = []
        batch_orgs = []
        batch_ids = []
        for idx, row in df.iterrows():
            interval = genome.Interval(
                chromosome=row['chromosome'],
                start=int(row['start']),
                end=int(row['end']),
            )
            interval = interval.resize(sequence_length)
            try:
                seq_str = fasta_extractor.extract(interval)
                seq_one_hot = one_hot_enc.encode(seq_str)
                batch_seqs.append(seq_one_hot)
                batch_orgs.append(0)  # HOMO_SAPIENS
                batch_ids.append(row['ensembl_transcript_id'])
            except Exception as e:
                print(f"WARNUNG: Fehler beim Extrahieren der Sequenz an Index {idx} (Transcript ID: {row.get('ensembl_transcript_id', 'Unbekannt')}): {e}")
                continue

            if len(batch_seqs) == args.batch_size:
                yield schemas.DataBatch(
                    dna_sequence=np.array(batch_seqs),
                    organism_index=np.array(batch_orgs),
                    rna_half_life=np.zeros(args.batch_size, dtype=np.float32)
                ), batch_ids
                batch_seqs = []
                batch_orgs = []
                batch_ids = []
                
        if batch_seqs:
            yield schemas.DataBatch(
                dna_sequence=np.array(batch_seqs),
                organism_index=np.array(batch_orgs),
                rna_half_life=np.zeros(len(batch_seqs), dtype=np.float32)
            ), batch_ids

    all_embeddings = []
    all_ids = []
    print("Extrahiere Embeddings...")
    num_batches = (num_samples + args.batch_size - 1) // args.batch_size
    for batch, batch_ids in tqdm(get_batches(), total=num_batches):
        emb = embed_step(pretrained_params, pretrained_state, batch)
        all_embeddings.append(np.array(emb))
        all_ids.extend(batch_ids)
        
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_ids = np.array(all_ids)
    print(f"Insgesamt {all_embeddings.shape[0]} Embeddings der Form {all_embeddings.shape[1:]} extrahiert.")
    
    # Falls output_path auf .npy endet, ändern wir es zu .npz
    output_path = args.output_path
    if output_path.endswith('.npy'):
        output_path = output_path[:-4] + '.npz'
        print(f"Speicherpfad von .npy zu .npz geändert: {output_path}")

    # Ordner erstellen, falls nicht vorhanden
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    np.savez(output_path, embeddings=all_embeddings, ensembl_transcript_ids=all_ids)
    print(f"Embeddings und ensembl_transcript_ids erfolgreich unter {output_path} gespeichert.")

if __name__ == "__main__":
    main()
