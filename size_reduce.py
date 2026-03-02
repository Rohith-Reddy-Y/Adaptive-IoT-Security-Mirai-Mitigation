import pandas as pd
import os

# Input file
input_path = "browser_input_datasets/dataset12_part_0000_raw_clean.csv"

# Load CSV
df = pd.read_csv(input_path)

# Keep only first 510000 rows
df_trimmed = df.iloc[:810000]

# Create output file name
base_name = os.path.splitext(os.path.basename(input_path))[0]
output_path = f"browser_input_datasets/{base_name}_810000_rows.csv"

# Save trimmed file
df_trimmed.to_csv(output_path, index=False)

print(f"Saved first 510000 rows to {output_path}")