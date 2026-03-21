import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    def __init__(self, target_col="CLAIM_PAID"):
        self.target_col = target_col
        self.numeric_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.numeric_cols = []
        self.categorical_cols = []
        self.feature_names = []

    def _identify_column_types(self, df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if self.target_col in numeric_cols: numeric_cols.remove(self.target_col)
        if self.target_col in categorical_cols: categorical_cols.remove(self.target_col)

        return numeric_cols, categorical_cols

    def _handle_outliers(self, df, col):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[col].clip(lower=lower_bound, upper=upper_bound)

    def fit(self, df):
        self.numeric_cols, self.categorical_cols = self._identify_column_types(df)

        if self.numeric_cols: self.numeric_imputer.fit(df[self.numeric_cols])

        if self.categorical_cols: self.categorical_imputer.fit(df[self.categorical_cols])

        for col in self.categorical_cols:
            le = LabelEncoder()
            col_data = df[col].fillna('MISSING')
            le.fit(col_data.astype(str))
            self.label_encoders[col] = le

        if self.numeric_cols:
            numeric_imputed = self.numeric_imputer.transform(df[self.numeric_cols])
            self.scaler.fit(numeric_imputed)

        self.feature_names = self.numeric_cols + self.categorical_cols
        return self

    def transform(self, df):
        df_transformed = df.copy()

        for col in self.numeric_cols:
            if col in df_transformed.columns: df_transformed[col] = self._handle_outliers(df_transformed, col)

        if self.numeric_cols:
            numeric_data = df_transformed[self.numeric_cols]
            numeric_imputed = np.asarray(self.numeric_imputer.transform(numeric_data))
            df_transformed[self.numeric_cols] = pd.DataFrame(
                numeric_imputed,
                columns=self.numeric_cols,
                index=df_transformed.index
            )

        if self.categorical_cols:
            categorical_data = df_transformed[self.categorical_cols]
            categorical_imputed = np.asarray(self.categorical_imputer.transform(categorical_data))

            for i, col in enumerate(self.categorical_cols):
                if col in self.label_encoders:
                    col_data = pd.Series(categorical_imputed[:, i]).fillna('MISSING')
                    encoded_values = self.label_encoders[col].transform(col_data.astype(str))
                    df_transformed[col] = pd.Series(encoded_values, index=df_transformed.index)

        if self.numeric_cols:
            numeric_data = df_transformed[self.numeric_cols]
            scaled_values = self.scaler.transform(numeric_data.values)
            df_transformed[self.numeric_cols] = pd.DataFrame(scaled_values, columns=self.numeric_cols, index=df_transformed.index)

        return df_transformed

    def fit_transform(self, df):
        return self.fit(df).transform(df)


def clean_data(df,max_missing_rate = 0.6, max_duplicate_rate = 0.2, apply_rules = False, rules_miner = None):
    initial_rows = len(df)
    initial_cols = len(df.columns)

    missing_rates = df.isna().mean()
    cols_to_drop = missing_rates[missing_rates > max_missing_rate].index.tolist()

    if cols_to_drop: df = df.drop(columns=cols_to_drop)

    duplicate_rate = df.duplicated().mean()
    if duplicate_rate > max_duplicate_rate: df = df.drop_duplicates()

    df = df.dropna(how='all')
    cleaning_report = {
        'initial_rows': initial_rows,
        'initial_cols': initial_cols,
        'final_rows': len(df),
        'final_cols': len(df.columns),
        'rows_removed': initial_rows - len(df),
        'cols_removed': initial_cols - len(df.columns),
        'duplicate_rate': float(duplicate_rate),
        'dropped_columns': cols_to_drop,
        'rules_validation': None
    }

    if apply_rules and rules_miner:
        validation_report = rules_miner.validate_data(df)
        cleaning_report['rules_validation'] = validation_report

    return df