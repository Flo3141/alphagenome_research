import numpy as np

def load_embeddings():
    data = np.load("/beegfs/prj/RNA_NLP/AlphaGenome/embeddings/embeddings.npz")
    embeddings = data["embeddings"]
    transcript_ids = data["ensembl_transcript_ids"]
    half_lives = data["half_lives"]

    print("Embeddings Shape:", embeddings.shape)
    print("Transcript IDs Shape:", transcript_ids.shape)
    print("Half Lives Shape:", half_lives.shape)

    print(transcript_ids[0])
    print(half_lives[0])
    print(embeddings[0])

if __name__ == "__main__":
    # main()
    load_embeddings()