import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime


class AutoEDA:
    def __init__(self, output_dir: str = "reports/eda"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report = {}

    def analyze(self, df: pd.DataFrame, target_col: Optional[str] = None):
        self.report = {
            "timestamp": datetime.now().isoformat(),
            "basic_info": self._basic_info(df),
            "missing_values": self._missing_analysis(df),
            "numeric_stats": self._numeric_statistics(df),
            "categorical_stats": self._categorical_statistics(df),
            "correlations": self._correlation_analysis(df, target_col),
            "outliers": self._outlier_detection(df),
            "data_quality": self._data_quality_score(df),
        }

        if target_col and target_col in df.columns:
            self.report["target_analysis"] = self._target_analysis(df, target_col)

        return self.report

    def _basic_info(self, df: pd.DataFrame):
        return {
            "n_rows": int(df.shape[0]),
            "n_cols": int(df.shape[1]),
            "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024**2),
            "duplicates": int(df.duplicated().sum()),
            "duplicate_rate": float(df.duplicated().mean()),
        }

    def _missing_analysis(self, df: pd.DataFrame) -> Dict:
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)

        return {
            "total_missing": int(missing.sum()),
            "columns_with_missing": int((missing > 0).sum()),
            "missing_by_column": {
                col: {
                    "count": int(missing[col]),
                    "percentage": float(missing_pct[col])
                }
                for col in df.columns if missing[col] > 0
            }
        }

    def _numeric_statistics(self, df: pd.DataFrame):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        stats = {}

        for col in numeric_cols:
            data = df[col].dropna()
            if len(data) > 0:
                stats[col] = {
                    "mean": float(data.mean()),
                    "median": float(data.median()),
                    "std": float(data.std()),
                    "min": float(data.min()),
                    "max": float(data.max()),
                    "q25": float(data.quantile(0.25)),
                    "q75": float(data.quantile(0.75)),
                    "skewness": float(data.skew()),
                    "kurtosis": float(data.kurtosis()),
                }

        return stats

    def _categorical_statistics(self, df: pd.DataFrame):
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        stats = {}

        for col in cat_cols:
            value_counts = df[col].value_counts()
            stats[col] = {
                "unique_values": int(df[col].nunique()),
                "most_common": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                "most_common_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                "top_5_values": {
                    str(k): int(v) for k, v in value_counts.head(5).items()
                }
            }

        return stats

    def _correlation_analysis(self, df: pd.DataFrame, target_col: Optional[str]):
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.shape[1] < 2:
            return {}

        corr_matrix = numeric_df.corr()

        result = {
            "high_correlations": []
        }

        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    result["high_correlations"].append({
                        "feature1": corr_matrix.columns[i],
                        "feature2": corr_matrix.columns[j],
                        "correlation": float(corr_val)
                    })

        if target_col and target_col in numeric_df.columns:
            target_corr = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)
            result["target_correlations"] = {
                str(k): float(v) for k, v in target_corr.head(10).items()
            }

        return result

    def _outlier_detection(self, df: pd.DataFrame):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers = {}

        for col in numeric_cols:
            data = df[col].dropna()
            if len(data) > 0:
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outlier_count = ((data < lower_bound) | (data > upper_bound)).sum()

                if outlier_count > 0:
                    outliers[col] = {
                        "count": int(outlier_count),
                        "percentage": float(outlier_count / len(data) * 100),
                        "lower_bound": float(lower_bound),
                        "upper_bound": float(upper_bound)
                    }

        return outliers

    def _target_analysis(self, df: pd.DataFrame, target_col: str):
        target = df[target_col]

        analysis = {
            "type": "numeric" if pd.api.types.is_numeric_dtype(target) else "categorical",
            "missing_count": int(target.isnull().sum()),
            "missing_percentage": float(target.isnull().mean() * 100),
        }

        if pd.api.types.is_numeric_dtype(target):
            target_clean = target.dropna()
            if len(target_clean) > 0:
                analysis.update({
                    "mean": float(target_clean.mean()),
                    "median": float(target_clean.median()),
                    "std": float(target_clean.std()),
                    "min": float(target_clean.min()),
                    "max": float(target_clean.max()),
                    "zeros_count": int((target_clean == 0).sum()),
                    "zeros_percentage": float((target_clean == 0).mean() * 100),
                })
        else:
            value_counts = target.value_counts()
            analysis.update({
                "unique_values": int(target.nunique()),
                "distribution": {str(k): int(v) for k, v in value_counts.head(10).items()}
            })

        return analysis

    def _data_quality_score(self, df: pd.DataFrame):
        missing_score = 100 - (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
        duplicate_score = 100 - (df.duplicated().sum() / len(df) * 100)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        completeness_scores = []

        for col in df.columns:
            completeness = (1 - df[col].isnull().mean()) * 100
            completeness_scores.append(completeness)

        avg_completeness = np.mean(completeness_scores) if completeness_scores else 0

        overall_score = (missing_score * 0.4 + duplicate_score * 0.3 + avg_completeness * 0.3)

        return {
            "overall_score": float(overall_score),
            "missing_score": float(missing_score),
            "duplicate_score": float(duplicate_score),
            "completeness_score": float(avg_completeness),
            "quality_level": "excellent" if overall_score >= 90 else "good" if overall_score >= 75 else "fair" if overall_score >= 60 else "poor"
        }

    def save_report(self, filename: Optional[str] = None):
        if filename is None:
            filename = f"eda_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        report_path = self.output_dir / filename

        with open(report_path, 'w') as f:
            json.dump(self.report, f, indent=2)

        return report_path

    def generate_text_report(self):
        lines = []
        lines.append("=" * 80)
        lines.append("EXPLORATORY DATA ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {self.report.get('timestamp', 'N/A')}")
        lines.append("")

        basic = self.report.get('basic_info', {})
        lines.append("BASIC INFORMATION")
        lines.append("-" * 80)
        lines.append(f"Rows: {basic.get('n_rows', 0):,}")
        lines.append(f"Columns: {basic.get('n_cols', 0)}")
        lines.append(f"Memory Usage: {basic.get('memory_usage_mb', 0):.2f} MB")
        lines.append(f"Duplicates: {basic.get('duplicates', 0):,} ({basic.get('duplicate_rate', 0)*100:.2f}%)")
        lines.append("")

        quality = self.report.get('data_quality', {})
        lines.append("DATA QUALITY")
        lines.append("-" * 80)
        lines.append(f"Overall Score: {quality.get('overall_score', 0):.2f}/100")
        lines.append(f"Quality Level: {quality.get('quality_level', 'unknown').upper()}")
        lines.append(f"Missing Score: {quality.get('missing_score', 0):.2f}/100")
        lines.append(f"Duplicate Score: {quality.get('duplicate_score', 0):.2f}/100")
        lines.append("")

        missing = self.report.get('missing_values', {})
        if missing.get('columns_with_missing', 0) > 0:
            lines.append("MISSING VALUES")
            lines.append("-" * 80)
            lines.append(f"Total Missing: {missing.get('total_missing', 0):,}")
            lines.append(f"Columns with Missing: {missing.get('columns_with_missing', 0)}")
            lines.append("")
            for col, info in list(missing.get('missing_by_column', {}).items())[:10]:
                lines.append(f"  {col}: {info['count']:,} ({info['percentage']:.2f}%)")
            lines.append("")

        outliers = self.report.get('outliers', {})
        if outliers:
            lines.append("OUTLIERS DETECTED")
            lines.append("-" * 80)
            for col, info in list(outliers.items())[:10]:
                lines.append(f"  {col}: {info['count']:,} outliers ({info['percentage']:.2f}%)")
            lines.append("")

        if 'target_analysis' in self.report:
            target = self.report['target_analysis']
            lines.append("TARGET VARIABLE ANALYSIS")
            lines.append("-" * 80)
            lines.append(f"Type: {target.get('type', 'unknown')}")
            lines.append(f"Missing: {target.get('missing_count', 0):,} ({target.get('missing_percentage', 0):.2f}%)")
            if target.get('type') == 'numeric':
                lines.append(f"Mean: {target.get('mean', 0):.2f}")
                lines.append(f"Median: {target.get('median', 0):.2f}")
                lines.append(f"Std: {target.get('std', 0):.2f}")
                lines.append(f"Range: [{target.get('min', 0):.2f}, {target.get('max', 0):.2f}]")
            lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)

    def save_text_report(self, filename: Optional[str] = None):
        if filename is None:
            filename = f"eda_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        report_path = self.output_dir / filename

        with open(report_path, 'w') as f:
            f.write(self.generate_text_report())

        return report_path


def quick_eda(df: pd.DataFrame, target_col: Optional[str] = None, save: bool = True):
    eda = AutoEDA()
    report = eda.analyze(df, target_col)

    if save:
        eda.save_report()
        eda.save_text_report()

    print(eda.generate_text_report())

    return report
