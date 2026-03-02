import pandas as pd
import os

# Input file (inside browser_input_datasets folder)
input_path = "browser_input_datasets/dataset12_part_0000_raw_clean_810000_rows.csv"

# Load CSV file
df = pd.read_csv(input_path)

# Shuffle rows randomly
df_shuffled = df.sample(frac=1).reset_index(drop=True)

# Create output folder
output_folder = "shuffled_csv's"
os.makedirs(output_folder, exist_ok=True)

# Create dynamic output filename
base_name = os.path.splitext(os.path.basename(input_path))[0]
output_path = os.path.join(output_folder, f"{base_name}_shuffled.csv")

# Save shuffled CSV
df_shuffled.to_csv(output_path, index=False)

print(f"Rows shuffled successfully! Saved as {output_path}")