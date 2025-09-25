import pandas as pd
import streamlit as st
from wordcloud import WordCloud
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import matplotlib.pyplot as plt

def missing_val(df):
    # ---------------- CSS ----------------
    st.markdown("""
        <style>
        .stApp {
            font-family: "Segoe UI", sans-serif;
        }
        h2, h3, h4 {
            color: #333333;
        }
        .stSelectbox label {
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)
    # -------------------------------------

    duplicate_df = df.copy()
    temp_df = df.columns[df.isnull().any()].tolist()
    temp_df = df[temp_df]
    list_cols = temp_df.columns

    st.subheader("⚠️ Missing Value Handler")
    opt = st.selectbox("📌 Select a column with missing values", options=list_cols)

    # ---------------- Categorical columns ----------------
    if df[opt].dtype == 'object' or df[opt].dtype == 'category':
        duplicate_df[opt].fillna(duplicate_df[opt].mode()[0], inplace=True)
        st.success(f"✅ Missing values in **{opt}** column replaced with **mode**")

        if df[opt].nunique() < df[opt].shape[0] * 0.05:
            fig, ax = plt.subplots(figsize=(12, 10))
            df[opt].value_counts().plot(kind="bar", ax=ax, color="green")
            ax.set_title(f"📊 Frequency Distribution after Mode Imputation: {opt}")
            st.pyplot(fig)
            return 0

        fig, ax = plt.subplots(figsize=(12, 10))
        text = " ".join(df[opt].astype(str))
        wordcloud = WordCloud(width=400, height=200, background_color='white').generate(text)
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

    # ---------------- Numerical columns ----------------
    elif pd.api.types.is_numeric_dtype(df[opt]):
        curr = st.selectbox("⚙️ Choose imputation method", options=["None","Mean","Mode","Median","Iterative Imputer"])
        
        if curr == "Mean":
            duplicate_df[opt].fillna(duplicate_df[opt].mean(), inplace=True)
            st.success(f"✅ Missing values in **{opt}** replaced with **mean**")
            fig, ax = plt.subplots(figsize=(12,10))
            ax.hist(duplicate_df[opt], bins=15, color='skyblue', edgecolor='black')
            ax.set_title(f"📊 Histogram after Mean: {opt}")
            st.pyplot(fig)

        elif curr == "Median":
            duplicate_df[opt].fillna(duplicate_df[opt].median(), inplace=True)
            st.success(f"✅ Missing values in **{opt}** replaced with **median**")
            fig, ax = plt.subplots(figsize=(12,10))
            ax.hist(duplicate_df[opt], bins=15, color='skyblue', edgecolor='black')
            ax.set_title(f"📊 Histogram after Median: {opt}")
            st.pyplot(fig)

        elif curr == "Mode":
            duplicate_df[opt].fillna(duplicate_df[opt].mode()[0], inplace=True)
            st.success(f"✅ Missing values in **{opt}** replaced with **mode**")
            fig, ax = plt.subplots(figsize=(12,10))
            ax.hist(duplicate_df[opt], bins=15, color='skyblue', edgecolor='black')
            ax.set_title(f"📊 Histogram after Mode: {opt}")
            st.pyplot(fig)

        elif curr == 'Iterative Imputer':
            imputer = IterativeImputer(random_state=42, max_iter=10)
            df_imputed_array = imputer.fit_transform(duplicate_df[[opt]])
            df_imputed = pd.DataFrame(df_imputed_array, columns=[opt])
            
            st.success(f"✅ Missing values in **{opt}** handled using **Iterative Imputer**")
            fig, ax = plt.subplots(figsize=(12,10))
            ax.hist(df_imputed[opt], bins=15, color='skyblue', edgecolor='black')
            ax.set_title(f"📊 Histogram after Iterative Imputer: {opt}")
            st.pyplot(fig)

        else:
            st.info("ℹ️ No imputation applied")
