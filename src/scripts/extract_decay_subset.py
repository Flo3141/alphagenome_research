"""Script to extract a subset of columns from RNAdecayCafe_v1.1_onetable.csv."""

import argparse
import os
import pandas as pd


def extract_columns(
    input_path: str,
    output_path: str,
):
  """Reads the input CSV, extracts/computes target columns, and writes to output."""
  print(f"Reading input file: {input_path}")
  if not os.path.exists(input_path):
    raise FileNotFoundError(f"Input file not found at: {input_path}")

  # Check columns first to see if avg_donorm_halflife is already present
  # We read first 0 rows just to get the headers
  headers = pd.read_csv(input_path, nrows=0).columns.tolist()

  target_cols = [
      "cell_line",
      "seqnames",
      "strand",
      "start",
      "end",
  ]

  if "avg_donorm_halflife" in headers:
    print("Found 'avg_donorm_halflife' column in the input file.")
    usecols = target_cols + ["avg_donorm_halflife"]
    df = pd.read_csv(input_path, usecols=usecols)
    df = df[usecols]  # reorder
  elif "donorm_halflife" in headers:
    print(
        "Column 'avg_donorm_halflife' not found, but 'donorm_halflife' exists. "
        "Computing the average (mean) grouped by cell line and coordinates..."
    )
    usecols = target_cols + ["donorm_halflife"]
    df = pd.read_csv(input_path, usecols=usecols)
    
    # Group by cell line and genomic coordinates, then calculate the average
    df = (
        df.groupby(target_cols)["donorm_halflife"]
        .mean()
        .reset_index(name="avg_donorm_halflife")
    )
  else:
    raise ValueError(
        "Could not find 'avg_donorm_halflife' or 'donorm_halflife' columns in the CSV. "
        f"Available columns: {headers}"
    )

  print(f"Writing extracted/computed data ({len(df)} rows) to: {output_path}")
  df = df.rename(columns={"seqnames": "chromosome"})
  df.to_csv(output_path, index=False)
  print("Extraction complete successfully.")


def get_unique_cell_lines(csv_path: str) -> list[str]:
  """Reads a CSV file and returns a sorted list of unique cell lines."""
  print(f"Reading cell lines from: {csv_path}")
  df = pd.read_csv(csv_path, usecols=["cell_line"])
  return sorted(df["cell_line"].unique().tolist())


def main():
  parser = argparse.ArgumentParser(
      description="Extract specific columns from RNA decay CSV."
  )
  parser.add_argument(
      "--input",
      type=str,
      default=r"/beegfs/prj/RNA_NLP/AlphaGenome/data/RNAdecayCafe_v1.1_onetable.csv",
      help="Path to the input CSV file.",
  )
  parser.add_argument(
      "--output",
      type=str,
      default=r"/beegfs/prj/RNA_NLP/AlphaGenome/data/RNAdecayCafe_for_alphagenome.csv",
      help="Path to the output CSV file.",
  )
  parser.add_argument(
      "--list-cell-lines",
      action="store_true",
      help="Print all unique cell lines in the input CSV and exit.",
  )

  args = parser.parse_args()

  if args.list_cell_lines:
    try:
      cell_lines = get_unique_cell_lines(args.input)
      print(f"Unique cell lines ({len(cell_lines)}):")
      for cell_line in cell_lines:
        print(f"  - {cell_line}")
    except Exception as e:
      print(f"Error: {e}")
  else:
    extract_columns(args.input, args.output)


if __name__ == "__main__":
  main()
