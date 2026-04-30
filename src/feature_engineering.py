import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime


class FeatureEngineer:
    def __init__(self):
        self.feature_names = []
        self.is_fitted = False

    def create_temporal_features(self, df):
        df = df.copy()

        if 'INSR_BEGIN' in df.columns:
            try:
                df['INSR_BEGIN_dt'] = pd.to_datetime(df['INSR_BEGIN'], errors='coerce')
                df['INSR_BEGIN_year'] = df['INSR_BEGIN_dt'].dt.year
                df['INSR_BEGIN_month'] = df['INSR_BEGIN_dt'].dt.month
                df['INSR_BEGIN_quarter'] = df['INSR_BEGIN_dt'].dt.quarter
                df = df.drop('INSR_BEGIN_dt', axis=1)
            except:
                pass

        if 'INSR_END' in df.columns:
            try:
                df['INSR_END_dt'] = pd.to_datetime(df['INSR_END'], errors='coerce')
                df['INSR_END_year'] = df['INSR_END_dt'].dt.year
                df['INSR_END_month'] = df['INSR_END_dt'].dt.month
                df = df.drop('INSR_END_dt', axis=1)
            except:
                pass

        if 'INSR_BEGIN' in df.columns and 'INSR_END' in df.columns:
            try:
                begin = pd.to_datetime(df['INSR_BEGIN'], errors='coerce')
                end = pd.to_datetime(df['INSR_END'], errors='coerce')
                df['insurance_duration_days'] = (end - begin).dt.days
            except:
                pass

        return df

    def create_vehicle_features(self, df):
        df = df.copy()

        if 'PROD_YEAR' in df.columns and 'EFFECTIVE_YR' in df.columns:
            effective_yr_numeric = pd.to_numeric(df['EFFECTIVE_YR'], errors='coerce')
            prod_year_numeric = pd.to_numeric(df['PROD_YEAR'], errors='coerce')
            df['vehicle_age'] = effective_yr_numeric - prod_year_numeric
            df['vehicle_age'] = df['vehicle_age'].fillna(0).clip(lower=0)

        if 'PROD_YEAR' in df.columns:
            current_year = datetime.now().year
            prod_year_numeric = pd.to_numeric(df['PROD_YEAR'], errors='coerce')
            df['years_since_production'] = current_year - prod_year_numeric
            df['years_since_production'] = df['years_since_production'].fillna(0)
            df['is_new_vehicle'] = (df['years_since_production'] <= 1).astype(int)
            df['is_old_vehicle'] = (df['years_since_production'] >= 10).astype(int)

        if 'SEATS_NUM' in df.columns:
            df['is_small_vehicle'] = (df['SEATS_NUM'] <= 5).astype(int)
            df['is_large_vehicle'] = (df['SEATS_NUM'] > 7).astype(int)

        if 'CCM_TON' in df.columns:
            df['engine_size_category'] = pd.cut(df['CCM_TON'], bins=[0, 1500, 2500, 5000, np.inf],
                                                labels=[0, 1, 2, 3], include_lowest=True).astype(float)

        return df

    def create_financial_features(self, df):
        df = df.copy()

        if 'PREMIUM' in df.columns and 'INSURED_VALUE' in df.columns:
            df['premium_to_value_ratio'] = df['PREMIUM'] / (df['INSURED_VALUE'] + 1)
            df['value_per_premium'] = df['INSURED_VALUE'] / (df['PREMIUM'] + 1)

        if 'PREMIUM' in df.columns:
            df['log_premium'] = np.log1p(df['PREMIUM'])
            df['premium_squared'] = df['PREMIUM'] ** 2

        if 'INSURED_VALUE' in df.columns:
            df['log_insured_value'] = np.log1p(df['INSURED_VALUE'])
            df['is_high_value'] = (df['INSURED_VALUE'] > df['INSURED_VALUE'].median()).astype(int)

        if 'PREMIUM' in df.columns and 'vehicle_age' in df.columns:
            df['premium_per_age'] = df['PREMIUM'] / (df['vehicle_age'] + 1)

        return df

    def create_interaction_features(self, df):
        df = df.copy()

        if 'SEX' in df.columns and 'vehicle_age' in df.columns:
            df['sex_age_interaction'] = df['SEX'].astype(str) + '_' + pd.cut(df['vehicle_age'],
                                                                             bins=[0, 3, 7, 15, np.inf],
                                                                             labels=['new', 'mid', 'old', 'very_old']).astype(str)

        if 'INSR_TYPE' in df.columns and 'TYPE_VEHICLE' in df.columns:
            df['insurance_vehicle_type'] = df['INSR_TYPE'].astype(str) + '_' + df['TYPE_VEHICLE'].astype(str)

        return df

    def create_aggregation_features(self, df, group_cols = None):
        df = df.copy()

        if group_cols is None:
            group_cols = ['MAKE', 'TYPE_VEHICLE', 'USAGE']

        for col in group_cols:
            if col in df.columns and 'PREMIUM' in df.columns:
                agg_col_name = f'{col}_avg_premium'
                df[agg_col_name] = df.groupby(col)['PREMIUM'].transform('mean')

                if 'INSURED_VALUE' in df.columns:
                    agg_col_name = f'{col}_avg_value'
                    df[agg_col_name] = df.groupby(col)['INSURED_VALUE'].transform('mean')

        return df

    def create_risk_features(self, df):
        df = df.copy()

        risk_score = 0

        if 'vehicle_age' in df.columns:
            risk_score += (df['vehicle_age'] > 10).astype(int) * 2

        if 'SEATS_NUM' in df.columns:
            risk_score += (df['SEATS_NUM'] > 7).astype(int)

        if 'USAGE' in df.columns:
            commercial_usage = df['USAGE'].isin(['Commercial', 'Taxi', 'Rental'])
            risk_score += commercial_usage.astype(int) * 3

        if 'premium_to_value_ratio' in df.columns:
            high_ratio = df['premium_to_value_ratio'] > df['premium_to_value_ratio'].quantile(0.75)
            risk_score += high_ratio.astype(int)

        df['risk_score'] = risk_score
        df['is_high_risk'] = (risk_score >= 3).astype(int)

        return df

    def fit_transform(self, df, target_col = None):
        df = self.transform(df, target_col)
        self.is_fitted = True
        self.feature_names = [col for col in df.columns if col != target_col]
        return df

    def transform(self, df, target_col = None):
        df = df.copy()

        df = self.create_temporal_features(df)
        df = self.create_vehicle_features(df)
        df = self.create_financial_features(df)
        df = self.create_interaction_features(df)
        df = self.create_aggregation_features(df)
        df = self.create_risk_features(df)

        return df

    def get_feature_importance_summary(self, df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        summary = {
            'total_features': len(df.columns),
            'numeric_features': len(numeric_cols),
            'categorical_features': len(df.columns) - len(numeric_cols),
            'engineered_features': []
        }

        engineered_keywords = ['_ratio', '_per_', 'log_', '_squared', '_interaction',
                              '_avg_', 'risk_', 'is_', 'vehicle_age', 'duration']

        for col in df.columns:
            if any(keyword in col for keyword in engineered_keywords):
                summary['engineered_features'].append(col)

        summary['n_engineered'] = len(summary['engineered_features'])

        return summary


def apply_feature_engineering(df, target_col = None):
    fe = FeatureEngineer()
    return fe.fit_transform(df, target_col)
