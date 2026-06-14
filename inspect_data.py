import pandas as pd
import numpy as np
import os

def inspect_dataset(file_path):
    output_lines = []
    output_lines.append("=== DATASET INSPECTION REPORT ===")
    
    if not os.path.exists(file_path):
        output_lines.append(f"Error: File '{file_path}' not found.")
        write_report(output_lines)
        return

    output_lines.append(f"Analyzing: {file_path}")
    
    # 1. Load the first row to get column names
    print("Reading column names...")
    df_sample = pd.read_csv(file_path, nrows=5)
    cols = list(df_sample.columns)
    output_lines.append(f"Total columns found: {len(cols)}")
    output_lines.append(f"First 20 columns: {cols[:20]}")
    output_lines.append(f"Last 50 columns: {cols[-50:]}")
    
    # 2. Check for missing values in a sample/chunk and identify column data types
    print("Reading full dataset in chunks...")
    chunks = pd.read_csv(file_path, chunksize=2000)
    total_rows = 0
    null_counts = np.zeros(len(cols))
    dtypes_dict = {}
    
    # Analyze the target distribution if we find target columns
    # Candidate target column names: look for class, label, target, mule, fraud, or similar
    possible_targets = [c for c in cols if any(word in c.lower() for word in ['class', 'label', 'target', 'mule', 'fraud', 'suspicious'])]
    if not possible_targets:
        # Fallback: look at the last column
        possible_targets = [cols[-1]]
    
    target_col = possible_targets[0]
    output_lines.append(f"\nTarget candidates found: {possible_targets}")
    output_lines.append(f"Selected target column for analysis: '{target_col}'")
    
    target_counts = {}
    
    for chunk in chunks:
        total_rows += len(chunk)
        null_counts += chunk.isnull().sum().values
        
        # Accumulate target counts
        if target_col in chunk.columns:
            for val in chunk[target_col].dropna():
                target_counts[val] = target_counts.get(val, 0) + 1

    output_lines.append(f"Total rows: {total_rows}")
    
    # 3. Analyze Column Types
    numeric_cols = 0
    categorical_cols = 0
    other_cols = 0
    for col in cols:
        col_type = str(df_sample[col].dtype)
        if 'int' in col_type or 'float' in col_type:
            numeric_cols += 1
        elif 'object' in col_type or 'category' in col_type:
            categorical_cols += 1
        else:
            other_cols += 1
            
    output_lines.append(f"\nColumn type distribution:")
    output_lines.append(f"  - Numeric columns: {numeric_cols}")
    output_lines.append(f"  - Categorical/Object columns: {categorical_cols}")
    output_lines.append(f"  - Other columns: {other_cols}")
    
    # 4. Target Distribution
    output_lines.append(f"\nTarget '{target_col}' class distribution:")
    for val, count in target_counts.items():
        percentage = (count / total_rows) * 100
        output_lines.append(f"  - {val}: {count} ({percentage:.2f}%)")
        
    # 5. Missing values overview
    high_null_cols = []
    for col, null_c in zip(cols, null_counts):
        null_pct = (null_c / total_rows) * 100
        if null_pct > 50:
            high_null_cols.append((col, null_pct))
            
    output_lines.append(f"\nMissing values stats:")
    output_lines.append(f"  - Columns with >50% missing values: {len(high_null_cols)} of {len(cols)}")
    if high_null_cols:
        output_lines.append(f"  - Examples of high missing value columns (top 10):")
        for col, pct in sorted(high_null_cols, key=lambda x: x[1], reverse=True)[:10]:
            output_lines.append(f"    - {col}: {pct:.2f}% missing")
            
    write_report(output_lines)
    print("Report written to data_summary.txt")

def write_report(lines):
    with open("data_summary.txt", "w") as f:
        f.write("\n".join(lines))

if __name__ == "__main__":
    inspect_dataset("DataSet.csv")
