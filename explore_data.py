import streamlit as st 
import pandas as pd
from scipy import stats
from statsmodels.stats.weightstats import ztest
import numpy as np

def show(df):
    # ---------------- CSS ----------------
    st.markdown("""
        <style>
        .stApp {
            font-family: "Segoe UI", sans-serif;
        }
        .stDataFrame, .stTable {
            border: 1px solid #ddd;
            border-radius: 6px;
        }
        h2, h3, h4 {
            color: #333333;
        }
        </style>
    """, unsafe_allow_html=True)
    # -------------------------------------

    with st.container():
        st.subheader("📋 Sample Table Overview")
        st.dataframe(df.sample(7))
        
    with st.container():
        st.subheader("📊 Statistical Summary")
        st.dataframe(df.describe())
    
    with st.container():
        st.subheader("🔠 Data Types")
        for col in df.columns:
            if df[col].nunique() < df.shape[0] * 0.05:
                 df[col] = df[col].astype("category")

        dtypes_df = pd.DataFrame(df.dtypes, columns=['Data Type'])
        st.dataframe(dtypes_df)

    st.header("🧮 Statistical Tests")

    # ---------------- T-Test ----------------
    with st.container():
        st.subheader("📌 T-Tests")
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['category']).columns.tolist()
        results = []

        for num_col in numeric_cols:
            for cat_col in categorical_cols:
                if df[cat_col].nunique() == 2:  # only 2 groups
                    groups = df.groupby(cat_col)[num_col].apply(list)
                    t_stat, p_val = stats.ttest_ind(groups.iloc[0], groups.iloc[1], nan_policy='omit')
                    results.append([f"{num_col} ~ {cat_col}", round(t_stat, 2), round(p_val, 4)])

        tdf = pd.DataFrame(results, columns=['Comparison', 'T-Test Statistic', 'p-value'])
        st.table(tdf)

    # ---------------- Z-Test ----------------
    with st.container():
        st.subheader("📌 Z-Tests")
        zdf = pd.DataFrame(columns=['Comparison','Z-Test Statistic','p-value'])

        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):  # avoid duplicate & self comparisons
                z_stat, p_value = ztest(df[numeric_cols[i]], df[numeric_cols[j]])
                zdf.loc[len(zdf)] = [f"{numeric_cols[i]} vs {numeric_cols[j]}", np.round(z_stat, 2), np.round(p_value,4)]

        st.table(zdf)

    # ---------------- Chi-Square ----------------
    with st.container():
        st.subheader("📌 Chi-Squared Tests")
        cdf = pd.DataFrame(columns=['Comparison','Chi² Statistic','p-value'])

        for i in range(len(categorical_cols)):
            for j in range(i+1, len(categorical_cols)):  # avoid duplicate & self comparisons
                contingency_table = pd.crosstab(df[categorical_cols[i]], df[categorical_cols[j]])
                chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
                cdf.loc[len(cdf)] = [f"{categorical_cols[i]} vs {categorical_cols[j]}", round(chi2,2), round(p,4)]

        st.table(cdf)
