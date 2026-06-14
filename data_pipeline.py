import pandas as pd
import numpy as np
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
import joblib

warnings.filterwarnings('ignore')

class FraudDataCleaner(BaseEstimator, TransformerMixin):
    """
    Production-ready data cleaning pipeline for Banking Fraud Detection.
    Implements all requirements including handling high-cardinality,
    multicollinearity, target leakage, and structural noise.
    """
    
    def __init__(self, target_col='F3924', missing_threshold=0.95, corr_threshold=0.95, leakage_threshold=0.80):
        self.target_col = target_col
        self.missing_threshold = missing_threshold
        self.corr_threshold = corr_threshold
        self.leakage_threshold = leakage_threshold
        
        # State variables learned during fit
        self.drop_cols_ = set()
        self.imputation_values_ = {}
        self.categorical_cols_ = []
        self.numeric_cols_ = []
        self.date_cols_ = []
        self.onehot_categories_ = {}
        self.columns_with_missing_ = []
        
    def fit(self, X, y=None):
        print("Starting Pipeline Fit...")
        df = X.copy()
        
        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in dataset.")
            
        y = df[self.target_col]
        
        # 1. & 6a. High Missing & Entirely Empty
        print("1. Identifying high-missing and empty columns...")
        missing_pct = df.isnull().mean()
        high_missing = missing_pct[missing_pct > self.missing_threshold].index.tolist()
        self.drop_cols_.update([c for c in high_missing if c != self.target_col])
        print(f"   -> Dropping {len(high_missing)} columns with > {self.missing_threshold*100}% missing.")
        
        # Keep track of which columns (that we aren't dropping) have missing values for indicators
        remaining_cols = [c for c in df.columns if c not in self.drop_cols_]
        self.columns_with_missing_ = [c for c in remaining_cols if missing_pct[c] > 0 and missing_pct[c] <= self.missing_threshold]

        # 6b. Constant Columns
        print("2. Identifying constant columns...")
        constant_cols = [c for c in remaining_cols if df[c].nunique(dropna=True) <= 1]
        self.drop_cols_.update([c for c in constant_cols if c != self.target_col])
        print(f"   -> Dropping {len(constant_cols)} constant columns.")
        
        # 6c. Duplicate Columns
        print("3. Identifying duplicate columns (fast hash)...")
        remaining_cols = [c for c in df.columns if c not in self.drop_cols_]
        df_filled = df[remaining_cols].fillna(-99999) # Placeholder for hashing
        hashes = {}
        duplicates = []
        for c in remaining_cols:
            if c == self.target_col:
                continue
            try:
                col_hash = hash(tuple(df_filled[c].values))
                if col_hash in hashes:
                    duplicates.append(c)
                else:
                    hashes[col_hash] = c
            except Exception:
                pass
        self.drop_cols_.update(duplicates)
        print(f"   -> Dropping {len(duplicates)} duplicate columns.")
        
        # Identify DataTypes
        remaining_cols = [c for c in df.columns if c not in self.drop_cols_]
        for c in remaining_cols:
            if c == self.target_col:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                self.numeric_cols_.append(c)
            else:
                # 4. Check if Date
                try:
                    # Quick check on first non-null valid element
                    sample = df[c].dropna().head(1).iloc[0]
                    # If it looks like a date (e.g. 8-1-2011)
                    if isinstance(sample, str) and len(sample.split('-')) == 3 and sample.replace('-','').isdigit():
                        self.date_cols_.append(c)
                    else:
                        self.categorical_cols_.append(c)
                except:
                    self.categorical_cols_.append(c)
                    
        # 8. Learn Imputation values (Median / Mode)
        print("4. Learning imputation medians and modes...")
        for c in self.numeric_cols_:
            val = df[c].median()
            self.imputation_values_[c] = val if pd.notnull(val) else 0.0
            
        for c in self.categorical_cols_:
            val = df[c].mode()
            self.imputation_values_[c] = val[0] if len(val) > 0 else "Unknown"

        # 5. Detect Target Leakage (corr > 0.8 with target)
        print("5. Checking target leakage...")
        # Temp impute for correlation
        df_num_imp = df[self.numeric_cols_].fillna(df[self.numeric_cols_].median()).fillna(0)
        
        # Using numpy for speed
        if df_num_imp.shape[1] > 0:
            target_vals = y.values
            leakage_cols = []
            for i, c in enumerate(self.numeric_cols_):
                corr = np.abs(np.corrcoef(df_num_imp[c].values, target_vals)[0, 1])
                if corr > self.leakage_threshold:
                    leakage_cols.append(c)
            self.drop_cols_.update(leakage_cols)
            # Remove from numeric lists
            self.numeric_cols_ = [c for c in self.numeric_cols_ if c not in leakage_cols]
            print(f"   -> Dropping {len(leakage_cols)} leakage columns (corr > {self.leakage_threshold}).")
        
        # 7. Address Multicollinearity
        print("6. Checking multicollinearity...")
        df_num_imp = df[self.numeric_cols_].fillna(df[self.numeric_cols_].median()).fillna(0)
        if df_num_imp.shape[1] > 0:
            corr_matrix = np.abs(np.corrcoef(df_num_imp.values, rowvar=False))
            multicoll_drop = []
            for i in range(len(self.numeric_cols_)):
                for j in range(i + 1, len(self.numeric_cols_)):
                    if corr_matrix[i, j] > self.corr_threshold:
                        col2 = self.numeric_cols_[j]
                        if col2 not in multicoll_drop:
                            multicoll_drop.append(col2)
                            
            self.drop_cols_.update(multicoll_drop)
            self.numeric_cols_ = [c for c in self.numeric_cols_ if c not in multicoll_drop]
            print(f"   -> Dropping {len(multicoll_drop)} multicollinear columns (corr > {self.corr_threshold}).")

        # Learn Categorical OneHot representations (Keep top 10 most frequent per column to avoid explosion)
        for c in self.categorical_cols_:
            top_cats = df[c].value_counts().head(10).index.tolist()
            self.onehot_categories_[c] = top_cats

        print("Fit complete!")
        return self

    def transform(self, X):
        print("Starting Pipeline Transform...")
        df = X.copy()
        
        # 2. Create missing indicators
        print("   -> Creating missing value indicators...")
        for c in self.columns_with_missing_:
            if c in df.columns:
                df[f"{c}_is_missing"] = df[c].isnull().astype(int)
        
        # Drop noise, duplicates, high nulls, multicollinear, leakage
        print("   -> Dropping irrelevant columns...")
        cols_to_drop = [c for c in self.drop_cols_ if c in df.columns]
        df.drop(columns=cols_to_drop, inplace=True)
        
        # 8. Imputation
        print("   -> Imputing missing values...")
        for c in self.numeric_cols_:
            if c in df.columns:
                df[c] = df[c].fillna(self.imputation_values_[c])
                
        for c in self.categorical_cols_:
            if c in df.columns:
                df[c] = df[c].fillna(self.imputation_values_[c])
                
        # 4. Parse Date Features
        print("   -> Parsing date features...")
        for c in self.date_cols_:
            if c in df.columns:
                # Convert to datetime
                df[c] = pd.to_datetime(df[c], errors='coerce')
                # Extract features
                df[f"{c}_year"] = df[c].dt.year
                df[f"{c}_month"] = df[c].dt.month
                df[f"{c}_day"] = df[c].dt.day
                df[f"{c}_dayofweek"] = df[c].dt.dayofweek
                # Impute new date features with -1
                df.fillna({f"{c}_year": -1, f"{c}_month": -1, f"{c}_day": -1, f"{c}_dayofweek": -1}, inplace=True)
                # Drop original date string
                df.drop(columns=[c], inplace=True)
                
        # 3. Encode Categorical Features
        print("   -> One-Hot Encoding categorical features...")
        for c in self.categorical_cols_:
            if c in df.columns:
                # Only keep top categories learned during fit
                top_cats = self.onehot_categories_[c]
                # Map rare categories to 'Other'
                df[c] = df[c].apply(lambda x: x if x in top_cats else 'Other')
                # Create dummies
                dummies = pd.get_dummies(df[c], prefix=c, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                # Drop original categorical column
                df.drop(columns=[c], inplace=True)
                
        print(f"Transform complete! New shape: {df.shape}")
        return df

def run_pipeline():
    print("Loading raw dataset...")
    df = pd.read_csv("DataSet.csv")
    
    cleaner = FraudDataCleaner(
        target_col='F3924',
        missing_threshold=0.95,
        corr_threshold=0.95,
        leakage_threshold=0.80
    )
    
    # Fit and transform
    df_cleaned = cleaner.fit_transform(df)
    
    # 9. Save Cleaned Dataset
    output_file = "DataSet_Cleaned.csv"
    print(f"Saving cleaned dataset to {output_file}...")
    df_cleaned.to_csv(output_file, index=False)
    
    # Save the fitted pipeline model for production inference
    joblib.dump(cleaner, "fraud_data_cleaner_pipeline.pkl")
    print("Pipeline model saved to 'fraud_data_cleaner_pipeline.pkl' for production use.")
    print("DONE.")

if __name__ == "__main__":
    run_pipeline()
