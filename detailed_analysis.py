import pandas as pd
import numpy as np
import os
import json
import traceback

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTS_AVAILABLE = True
except ImportError:
    PLOTS_AVAILABLE = False

def run_detailed_analysis(file_path):
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    print(f"Dataset loaded. Shape: {df.shape}")
    
    report = {}
    
    # 1. Identify Data Types
    print("Identifying data types...")
    dtypes_counts = df.dtypes.value_counts()
    dtypes_dict = {str(k): int(v) for k, v in dtypes_counts.items()}
    report["data_types_summary"] = dtypes_dict
    
    # List columns that are not float or int
    non_numeric_cols = []
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric_cols.append({
                "column": col,
                "dtype": str(df[col].dtype),
                "nunique": int(df[col].nunique()),
                "sample_values": [str(x) for x in df[col].dropna().head(5).tolist()]
            })
    report["non_numeric_columns"] = non_numeric_cols
    
    # 2. Find Missing Values
    print("Analyzing missing values...")
    null_counts = df.isnull().sum()
    null_percentages = (null_counts / len(df)) * 100
    
    report["missing_values"] = {
        "total_missing_cells": int(df.isnull().sum().sum()),
        "percent_missing_cells": float((df.isnull().sum().sum() / df.size) * 100),
        "columns_with_any_missing": int((null_counts > 0).sum()),
        "columns_over_50_pct_missing": int((null_percentages > 50).sum()),
        "columns_100_pct_missing": int((null_percentages == 100).sum()),
        "top_missing_columns": [
            {"column": col, "percent": float(pct)} 
            for col, pct in null_percentages.sort_values(ascending=False).head(30).items()
        ]
    }
    
    # 3. Find Constant Columns (nunique == 1 or std == 0)
    print("Finding constant columns...")
    constant_cols = []
    for col in df.columns:
        if df[col].nunique(dropna=True) <= 1:
            constant_cols.append(col)
    report["constant_columns"] = {
        "count": len(constant_cols),
        "columns": constant_cols[:50],
        "total_columns_list": constant_cols
    }
    
    # 4. Find Duplicate Columns (using column hashing for speed)
    print("Finding duplicate columns...")
    # Fill NAs with a placeholder to compute hashes
    df_filled = df.fillna(-99999)
    hashes = {}
    duplicate_groups = []
    duplicate_cols_list = []
    
    for col in df.columns:
        try:
            col_hash = hash(tuple(df_filled[col].values))
            if col_hash in hashes:
                hashes[col_hash].append(col)
            else:
                hashes[col_hash] = [col]
        except Exception as e:
            pass
            
    for h, cols in hashes.items():
        if len(cols) > 1:
            duplicate_groups.append(cols)
            duplicate_cols_list.extend(cols[1:])
            
    report["duplicate_columns"] = {
        "count": len(duplicate_cols_list),
        "duplicate_groups": [g[:10] for g in duplicate_groups[:20]],
        "total_duplicates_list": duplicate_cols_list
    }
    
    # 5. Find Highly Correlated Features
    # To avoid memory issues and speed up, we filter out high-null columns first.
    print("Finding highly correlated features...")
    exclude_cols = set(constant_cols) | set(null_percentages[null_percentages > 50].index) | set(duplicate_cols_list)
    cols_to_corr = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
    
    print(f"Number of columns for correlation check after filtering: {len(cols_to_corr)}")
    
    # We will use numpy corrcoef which is extremely fast and efficient
    df_numeric_clean = df[cols_to_corr].fillna(df[cols_to_corr].median())
    
    # Drop any column that still has NaNs (e.g. all values were NaN)
    df_numeric_clean = df_numeric_clean.dropna(axis=1, how='any')
    cols_to_corr_final = list(df_numeric_clean.columns)
    
    # Compute correlation matrix using numpy
    print("Computing correlation matrix using numpy...")
    corr_matrix_np = np.corrcoef(df_numeric_clean.values, rowvar=False)
    corr_matrix_np = np.abs(corr_matrix_np)
    
    # Find features with correlation greater than 0.95
    to_drop = []
    high_corr_pairs = []
    
    # We only need upper triangle
    n_features = len(cols_to_corr_final)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            val = corr_matrix_np[i, j]
            if val > 0.95:
                col1 = cols_to_corr_final[i]
                col2 = cols_to_corr_final[j]
                high_corr_pairs.append({"feature1": col1, "feature2": col2, "correlation": float(val)})
                if col2 not in to_drop:
                    to_drop.append(col2)
                    
    report["correlated_features"] = {
        "high_corr_threshold": 0.95,
        "features_to_drop_count": len(to_drop),
        "features_to_drop_sample": to_drop[:50],
        "top_correlated_pairs": sorted(high_corr_pairs, key=lambda x: x["correlation"], reverse=True)[:50],
        "total_drop_list": to_drop
    }
    
    # 6. Target Leakage Detection
    print("Detecting target leakage...")
    target_col = "F3924"
    leakage_candidates = []
    
    if target_col in df.columns:
        # Calculate correlation of all numeric features with target
        # Using numpy for speed
        target_idx = cols_to_corr_final.index(target_col) if target_col in cols_to_corr_final else -1
        
        if target_idx != -1:
            for i, col in enumerate(cols_to_corr_final):
                if col == target_col:
                    continue
                corr_val = corr_matrix_np[i, target_idx]
                if corr_val > 0.8:
                    leakage_candidates.append({
                        "column": col,
                        "correlation_with_target": float(corr_val),
                        "reason": f"Extremely high correlation with target ({corr_val:.4f})"
                    })
        else:
            # Fallback: pandas corrwith
            corr_with_target = df_numeric_clean.corrwith(df[target_col]).abs()
            for col, corr_val in corr_with_target.items():
                if col != target_col and corr_val > 0.8:
                    leakage_candidates.append({
                        "column": col,
                        "correlation_with_target": float(corr_val),
                        "reason": f"Extremely high correlation with target ({corr_val:.4f})"
                    })
                    
        report["target_leakage"] = {
            "target_column": target_col,
            "leakage_candidates_count": len(leakage_candidates),
            "candidates": leakage_candidates
        }
    else:
        report["target_leakage"] = {"error": f"Target column '{target_col}' not found"}
        
    # Write report to a JSON file
    print("Writing analysis report to JSON...")
    with open("detailed_analysis_results.json", "w") as f:
        json.dump(report, f, indent=4)
        
    # Generate Visualizations
    if PLOTS_AVAILABLE:
        print("Generating visualizations...")
        try:
            # 1. Class Imbalance Plot
            plt.figure(figsize=(8, 5))
            target_counts = df[target_col].value_counts(dropna=False)
            sns.barplot(x=target_counts.index, y=target_counts.values, palette="viridis")
            plt.title("Target Class Imbalance ('F3924')")
            plt.xlabel("Mule Account Class (0 = Legitimate, 1 = Mule)")
            plt.ylabel("Number of Accounts")
            for i, val in enumerate(target_counts.values):
                pct_val = (val / len(df)) * 100
                plt.text(i, val + 100, f"{val} ({pct_val:.2f}%)", ha='center', va='bottom', fontsize=11)
            plt.tight_layout()
            plt.savefig("class_imbalance.png", dpi=300)
            plt.close()

            # 2. Missing Values Distribution Plot
            plt.figure(figsize=(10, 5))
            sns.histplot(null_percentages, bins=50, kde=True, color="skyblue")
            plt.title("Distribution of Missing Values Percentage per Feature")
            plt.xlabel("Percentage of Missing Values (%)")
            plt.ylabel("Number of Features")
            plt.tight_layout()
            plt.savefig("missing_values_dist.png", dpi=300)
            plt.close()

            # 3. Correlation Heatmap
            plt.figure(figsize=(12, 10))
            # Find top 15 correlated columns with target
            corr_with_target_all = df_numeric_clean.corrwith(df[target_col]).abs().sort_values(ascending=False)
            top_features = corr_with_target_all.head(16).index.tolist()
            top_corr_matrix = df_numeric_clean[top_features].corr()
            sns.heatmap(top_corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
            plt.title("Correlation Matrix of Target & Top 15 Highly Correlated Features")
            plt.tight_layout()
            plt.savefig("correlation_heatmap.png", dpi=300)
            plt.close()
            print("Visualizations saved as class_imbalance.png, missing_values_dist.png, and correlation_heatmap.png")
        except Exception as e:
            print(f"Error during visualization generation: {e}")

    # Write report to a text summary file
    print("Writing analysis summary to text file...")
    summary_lines = []
    summary_lines.append("=== DETAILED ANALYSIS SUMMARY ===")
    summary_lines.append(f"Dataset Shape: {df.shape}")
    summary_lines.append(f"Target Column: {target_col}")
    summary_lines.append(f"Class Distribution: {df[target_col].value_counts().to_dict()}")
    summary_lines.append("\n1. Data Types:")
    for dtype, count in dtypes_dict.items():
        summary_lines.append(f"  - {dtype}: {count}")
    summary_lines.append(f"  - Non-numeric columns: {len(non_numeric_cols)}")
    for nnc in non_numeric_cols:
        summary_lines.append(f"    * {nnc['column']} ({nnc['dtype']}): nunique={nnc['nunique']}, samples={nnc['sample_values']}")
        
    summary_lines.append("\n2. Missing Values:")
    summary_lines.append(f"  - Total missing cells: {report['missing_values']['total_missing_cells']} ({report['missing_values']['percent_missing_cells']:.2f}%)")
    summary_lines.append(f"  - Columns with any missing values: {report['missing_values']['columns_with_any_missing']}")
    summary_lines.append(f"  - Columns with >50% missing values: {report['missing_values']['columns_over_50_pct_missing']}")
    summary_lines.append(f"  - Columns with 100% missing values: {report['missing_values']['columns_100_pct_missing']}")
    
    summary_lines.append("\n3. Constant Columns:")
    summary_lines.append(f"  - Count: {report['constant_columns']['count']}")
    summary_lines.append(f"  - Examples (first 20): {report['constant_columns']['columns'][:20]}")
    
    summary_lines.append("\n4. Duplicate Columns:")
    summary_lines.append(f"  - Count: {report['duplicate_columns']['count']}")
    summary_lines.append(f"  - Examples of duplicate groups (first 5 groups):")
    for group in report['duplicate_columns']['duplicate_groups'][:5]:
        summary_lines.append(f"    * Group: {group}")
        
    summary_lines.append("\n5. Highly Correlated Features (threshold > 0.95):")
    summary_lines.append(f"  - Features to drop count: {report['correlated_features']['features_to_drop_count']}")
    summary_lines.append(f"  - Top 10 correlated pairs:")
    for pair in report['correlated_features']['top_correlated_pairs'][:10]:
        summary_lines.append(f"    * {pair['feature1']} <-> {pair['feature2']}: corr = {pair['correlation']:.4f}")
        
    summary_lines.append("\n6. Target Leakage Candidates:")
    summary_lines.append(f"  - Leakage candidates count: {report['target_leakage'].get('leakage_candidates_count', 0)}")
    if 'candidates' in report['target_leakage']:
        for cand in report['target_leakage']['candidates']:
            summary_lines.append(f"    * {cand['column']}: corr = {cand['correlation_with_target']:.4f} ({cand['reason']})")
            
    with open("detailed_analysis_summary.txt", "w") as f:
        f.write("\n".join(summary_lines))
        
    print("Analysis finished successfully!")

if __name__ == "__main__":
    try:
        run_detailed_analysis("DataSet.csv")
    except Exception as e:
        with open("error_log.txt", "w") as f:
            f.write(f"Error occurred: {e}\n")
            f.write(traceback.format_exc())
        print(f"Error written to error_log.txt: {e}")
