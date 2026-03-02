import pandas as pd
import os

# Input parquet file (inside parquet files folder)
input_path = "parquet_files/dataset12_part_0000.parquet"

# Load raw parquet
df = pd.read_parquet(input_path)

print("Original Columns:")
print(df.columns.tolist())

# Remove label columns if present
df = df.drop(columns=["label", "detailed_label"], errors="ignore")

print("\nColumns after removing labels:")
print(df.columns.tolist())

# Create output filename dynamically
base_name = os.path.splitext(os.path.basename(input_path))[0]
output_path = f"browser_input_datasets/{base_name}_raw_clean.csv"

# Save cleaned version in browser_input_datasets folder
df.to_csv(output_path, index=False)

print(f"\nSaved cleaned file as {output_path}")