import pandas as pd
import streamlit as st 
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- CSS Styling ----------------
st.markdown("""
    <style>
    .stApp {
        font-family: "Segoe UI", Tahoma, sans-serif;
        background-color: #f8f9fa;
    }
    h2, h3 {
        color: #2c3e50;
        font-weight: 600;
    }
    .stDataFrame table {
        border-radius: 6px;
    }
    </style>
""", unsafe_allow_html=True)
# --------------------------------------------

def var_analysis(df):
    st.markdown("## 🔍 Variable Analysis")
    for i in df.columns:
        temp_df = df[i]
        if pd.api.types.is_numeric_dtype(temp_df):
            numeric_col(temp_df)
        elif pd.api.types.is_object_dtype(temp_df):
            obj_col(temp_df)
        else:
            cat_col(temp_df)

def count_zeros(column: pd.Series) -> int:
    """Return the number of zeros in a Series."""
    return (column == 0).sum()

# ---------------- Numeric Column ----------------
def numeric_col(series):
    with st.container():
        st.subheader(f"📈 {series.name}")
        uqcount = series.nunique()
        nullcount = series.isnull().sum()
        
        summary_dict = {
            'metrics': [
                uqcount,
                round((uqcount / series.shape[0]) *100,2),
                nullcount,
                round((nullcount / series.shape[0]) * 100,2),
                count_zeros(series)
            ]
        }

        summary_dict2 = {
            "metric":[
                series.min(),
                series.max(),
                round(series.mean(),2),
                round(series.var(),2),
                series.memory_usage(deep=True)
            ]
        }

        summary_df = pd.DataFrame(summary_dict, index=['Distinct','Distinct (%)','Missing','Missing (%)',"Zeros"])
        summary_df2 = pd.DataFrame(summary_dict2,index=['Minimum','Maximum','Mean','Variance','Size (bytes)'])

        col1, col2, col3 = st.columns(3)

        with col1:
            st.dataframe(summary_df,height=250)

        with col2:
            st.dataframe(summary_df2,height=250)

        with col3:
            fig, ax = plt.subplots(figsize=(8,6))
            ax.hist(series.dropna(), bins=15, color='skyblue', edgecolor='black')
            ax.set_title(f"📊 Histogram / KDE of {series.name}")
            st.pyplot(fig)

# ---------------- Categorical Column ----------------
def cat_col(series):
    with st.container():
        st.subheader(f"📦 {series.name}")
        uqcount = series.nunique()
        nullcount = series.isnull().sum()
        dict_c = {"metric":[uqcount, round((uqcount/series.shape[0])*100,2), nullcount, round(nullcount/series.shape[0]*100,2), series.memory_usage(deep=True)]}
        dict_cc = pd.DataFrame(dict_c,index=['Distinct','Distinct(%)','Missing','Missing(%)','Memory Size'])

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(dict_cc,height=250)
        with col2:
            fig,ax = plt.subplots(figsize=(8,6))
            series.value_counts().plot(kind='bar', ax=ax, color='lightgreen', edgecolor='black')
            ax.set_title(f"📊 {series.name} Countplot")
            st.pyplot(fig)

# ---------------- Object Column ----------------
def obj_col(series):
    with st.container():
        st.subheader(f"📝 {series.name}")
        uqcount = series.nunique()
        nullcount = series.isnull().sum()
        dict_o = {"metric":[uqcount, round((uqcount/series.shape[0])*100,2), nullcount, round(nullcount/series.shape[0]*100,2), series.memory_usage(deep=True)]}
        dict_oo = pd.DataFrame(dict_o,index=['Distinct','Distinct(%)','Missing','Missing(%)','Memory Size'])

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(dict_oo,height=250)

        with col2:
            text = " ".join(series.astype(str))
            wordcloud = WordCloud(width=400, height=200, background_color='white').generate(text)
            fig, ax = plt.subplots(figsize=(8,6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
