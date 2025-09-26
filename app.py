import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from explore_data import show
from Variable_Analysis import var_analysis
from Missing_value import missing_val
from plot import ploting

# ---------------- CSS ----------------
st.markdown("""
    <style>
    /* Main app background */
    .stApp {
        background-color: #f9f9f9;
        font-family: "Segoe UI", sans-serif;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        padding: 15px;
        border-right: 1px solid #ddd;
    }

    /* Titles */
    h1, h2, h3 {
        color: #333333;
    }

    /* File uploader */
    .stFileUploader label {
        font-weight: bold;
        color: #444;
    }

    /* Toast message */
    [data-testid="stToast"] {
        background-color: #f0f0f0;
        border-radius: 8px;
    }

    /* Buttons */
    button[kind="secondary"] {
        background-color: #4CAF50 !important;
        color: white !important;
        border-radius: 8px !important;
    }

    /* Images */
    img {
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)
# -------------------------------------

st.title("📊 DataPulse - Data Analysis App")

with st.sidebar:
    selected = option_menu(
        menu_title="🗂️ Main Menu",
        options=["🏠 Home","🔍 Explore Data","📈 Variable Analysis","⚠️ Missing Values","📊 Visualizations"],
        menu_icon="cast",
        default_index=0,
        orientation="vertical"
    )

data = st.sidebar.file_uploader("📤 Upload your CSV file", type=["csv"], key="file_uploader")
sel = st.sidebar.checkbox("📌 Use Sample Dataset", value=False)

if sel==True:
    df=pd.read_csv("titanic.csv")
    for col in df.columns:
        if df[col].nunique() < df.shape[0] * 0.05 or pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype("category")
elif sel==False:
    if data is not None:
        df=pd.read_csv(data)
        for col in df.columns:
            if df[col].nunique() < df.shape[0] * 0.05:
                df[col] = df[col].astype("category")
        st.toast("✅ Successfully uploaded the dataset")
else :
    pass

if selected=="🏠 Home":
    st.header("👋 Welcome to DataPulse")
    st.write("""
    🚀 DataPulse is an interactive data analysis application built with Streamlit.  
    It allows you to upload your datasets and perform various analyses including data exploration, 
    variable analysis, missing value handling, and visualizations.
    """)
    st.image("https://tse4.mm.bing.net/th/id/OIP.9iEVMyblp3R4eH46LCsWYwHaDt?pid=Api&P=0&h=180")
    st.write("## ✨ Features")
    st.write("""
    - 📂 **Data Upload**: Upload CSV files for analysis.
    - 🔍 **Explore Data**: View dataset overview and summary statistics.
    - 📈 **Variable Analysis**: Analyze individual variables with detailed statistics.
    - ⚠️ **Missing Values**: Identify and handle missing data.
    - 📊 **Visualizations**: Create various plots to visualize data distributions and relationships.
    """)

if selected=="🔍 Explore Data":
    show(df)

elif selected=="📈 Variable Analysis":
    var_analysis(df)

elif selected=="⚠️ Missing Values":
    missing_val(df)

elif selected=="📊 Visualizations":
    ploting(df)

else :
    pass
