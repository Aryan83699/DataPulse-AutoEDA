import os
import pandas as pd
from ydata_profiling import ProfileReport




def compute_complexity_score(df):
    import numpy as np

    n_rows, n_cols = df.shape

    # Size score (30%)
    row_score = min(np.log1p(n_rows) / np.log1p(100_000), 1.0) * 100
    col_score = min(n_cols / 50, 1.0) * 100
    size_score = (row_score * 0.6 + col_score * 0.4)

    # Missing values score (25%)
    missing_pct = df.isnull().mean().mean() * 100
    missing_score = min(missing_pct / 30, 1.0) * 100

    # Correlation score (20%)
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] >= 2:
        corr_matrix = numeric_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        corr_score = min(upper.stack().mean() / 0.8, 1.0) * 100
    else:
        corr_score = 0.0

    # Data type diversity score (15%)
    unique_kinds = df.dtypes.apply(lambda x: x.kind).nunique()
    diversity_score = min((unique_kinds - 1) / 4, 1.0) * 100

    # Outlier score (10%)
    if numeric_df.shape[1] > 0:
        def col_outlier_pct(s):
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            return 0.0 if iqr == 0 else ((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).mean() * 100
        outlier_score = min(numeric_df.apply(col_outlier_pct).mean() / 10, 1.0) * 100
    else:
        outlier_score = 0.0

    score = round(min(max(
        size_score * 0.30 +
        missing_score * 0.25 +
        corr_score * 0.20 +
        diversity_score * 0.15 +
        outlier_score * 0.10
    , 0), 100), 1)

    category = "Simple" if score <= 40 else "Moderate" if score <= 70 else "Complex"

    breakdown = {
        "size": round(size_score, 1),
        "missing": round(missing_score, 1),
        "correlation": round(corr_score, 1),
        "diversity": round(diversity_score, 1),
        "outliers": round(outlier_score, 1),
    }

    return score, category, breakdown



def generate_eda_report(file_path, output_folder="static/reports"):
    """
    Generates a ydata-profiling report for a given dataset
    and saves it inside static/reports folder.
    """

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    filename = os.path.basename(file_path)

    # ------------------------
    # Load dataset
    # ------------------------
    if filename.endswith(".csv"):
        df = pd.read_csv(file_path, low_memory=False)
    elif filename.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format for profiling")

    # ------------------------
    # Generate profiling report
    # ------------------------
    profile = ProfileReport(
        df,
        title=f"{filename} report",
        explorative=True,
            html={
        "style": {
            "primary_color": "#3c83f6",
            "logo": "",
            "theme": "flatly"
        }
    },
    plot={
        "histogram": {
            "x_axis_labels": True
        },
        "correlation": {
            "cmap": "Blues",
            "bad": "#f5f7f8"
        },
        "missing": {
            "cmap": "Blues"
        },
        "cat_freq": {
            "colors": ["#3c83f6", "#60a5fa", "#93c5fd", "#bfdbfe"]
        }
    }
)

    report_filename = f"{filename}_report.html"
    report_path = os.path.join(output_folder, report_filename)

    profile.to_file(report_path)

    score, category, breakdown = compute_complexity_score(df)
    return report_filename, score, category, breakdown