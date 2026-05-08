import os
import pandas as pd
import gffutils
import sys

def main():
    # Get the data folder from environment variable or use current directory
    data_folder = os.environ.get("DATA_FOLDER", ".")
    ag_data_folder = os.environ.get("AG_DATA_FOLDER", ".")
    
    db_path = os.path.join(data_folder, "sorted.gtf.db")
    csv_path = os.path.join(data_folder, "half_life.csv")
    output_path = os.path.join(ag_data_folder, "half_life_with_coords.csv")

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
                'end': feature.end
            })
        except gffutils.exceptions.FeatureNotFoundError:
            print(f"Warning: Transcript {tx_id_base} not found in database.", file=sys.stderr)
            coords.append({
                'chromosome': None,
                'start': None,
                'end': None
            })

    # Convert coordinates to DataFrame and join
    coords_df = pd.DataFrame(coords)
    result_df = pd.concat([df, coords_df], axis=1)

    print(f"Saving result to {output_path}")
    result_df.to_csv(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
