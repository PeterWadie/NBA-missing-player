import pandas as pd
from pathlib import Path

# Define the input directory and output file
INPUT_DIR = Path("input-dataset")
OUTPUT_FILE = Path("input-dataset/combined_filtered.csv")

# Initialize an empty list to store filtered DataFrames
filtered_dfs = []

# Loop through the years and process each file
for year in range(2007, 2016):  # 2015 is inclusive
    csv_filename = INPUT_DIR / f"matchups-{year}.csv"
    
    # Read the CSV file
    if csv_filename.exists():
        df = pd.read_csv(csv_filename)
        
        # Filter rows where 'outcome' column equals 1
        df_filtered = df[df['outcome'] == 1]
        
        # Append filtered DataFrame to the list
        filtered_dfs.append(df_filtered)
    else:
        print(f"File not found: {csv_filename}")

# Combine all filtered DataFrames into one
if filtered_dfs:
    combined_df = pd.concat(filtered_dfs, ignore_index=True)
    
    # Ensure output directory exists
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the combined filtered dataset to CSV
    combined_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Filtered data saved to {OUTPUT_FILE}")
else:
    print("No valid data found to save.")
