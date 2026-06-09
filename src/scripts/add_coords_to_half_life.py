import os
import pandas as pd
import gffutils
import sys

def create_half_life_csv():
    df = pd.read_csv("/beegfs/prj/RNA_NLP/RNA_half_lives/SalukiStyle_splits_sequence_transcript-estimates-annotated-pulseRTc-0-1-2-4-6-8-16_hIPSC_CM.txt", sep="\t", header=None)
    df.columns = ["data_split","ensembl_transcript_id","ensembl_gene_id","entrezgene_id","hgnc_symbol","transcript_biotype","rate","half_life","rate.min","rate.max","sequence"]
    half_life_df = df[["ensembl_transcript_id","half_life","rate","rate.min","rate.max"]]
    half_life_df.set_index("ensembl_transcript_id", inplace=True)
    half_life_df.to_csv(os.path.join(AG_DATA_FOLDER, "half_life.csv"), sep=",")

def main():
    # Get the data folder from environment variable or use current directory
    
    db_path = os.path.join(AG_DATA_FOLDER, "Homo_sapiens.GRCh38.108.gtf.db")
    csv_path = os.path.join(AG_DATA_FOLDER, "half_life.csv")
    output_path = os.path.join(AG_DATA_FOLDER, "half_life_with_coords.csv")

    if not os.path.exists(db_path):
        print(f"Error: GTF database not found at {db_path}")
        return

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    print(f"Loading GTF database from {db_path}...")
    db = gffutils.FeatureDB(db_path)

    print(f"Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Required columns in input CSV
    if 'ensembl_transcript_id' not in df.columns:
        print(f"Error: 'ensembl_transcript_id' column not found in {csv_path}")
        return

    coords = []
    print("Extracting coordinates from GTF database...")
    res = db.conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
    print(res.fetchall())
    columns = [info[1] for info in db.conn.execute("PRAGMA table_info(features);")]
    print(columns)

    for idx, row in df.iterrows():
        tx_id = row['ensembl_transcript_id']
        # Strip version suffix if present (e.g., ENST00000123.1 -> ENST00000123)
        tx_id_base = tx_id.split('.')[0]
        
        try:
            feature = db[tx_id_base]
            coords.append({
                'chromosome': "chr" + str(feature.seqid),
                'start': feature.start,
                'end': feature.end,
                'strand': feature.strand
            })
        except gffutils.exceptions.FeatureNotFoundError:
            print(f"Warning: Transcript {tx_id_base} not found in database.", file=sys.stderr)
            coords.append({
                'chromosome': None,
                'start': None,
                'end': None,
                'strand': None
            })

    # Convert coordinates to DataFrame and join
    coords_df = pd.DataFrame(coords)
    result_df = pd.concat([df, coords_df], axis=1)

    print(f"Saving result to {output_path}")
    result_df.to_csv(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    AG_DATA_FOLDER = os.environ.get("AG_DATA_FOLDER", ".")
    create_half_life_csv()
    main()
