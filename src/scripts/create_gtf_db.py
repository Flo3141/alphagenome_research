import gffutils
import os

if __name__ == "__main__":
    # Muss nur einmal ausgeführt werden
    print("Erstelle GFF-Datenbank (kann dauern)...")
    ag_data_folder = os.environ.get("AG_DATA_FOLDER", ".")
    db = gffutils.create_db(
        "/beegfs/prj/RNA_NLP/AlphaGenome/data/Homo_sapiens.GRCh38.115.gtf",
        dbfn=os.path.join(ag_data_folder, "Homo_sapiens.GRCh38.115.gtf.db"),
        force=True,
        keep_order=True,
        merge_strategy='merge',
        sort_attribute_values=True,
        disable_infer_genes=True,
        disable_infer_transcripts=True
    )
    print("Datenbank erstellt.")

    # Datenbank laden (für spätere Skripte)
    # db = gffutils.FeatureDB('annotations.db')