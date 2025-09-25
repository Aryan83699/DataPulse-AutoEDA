import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud

# Load your dataset
df = pd.read_csv("titanic.csv")

st.title("📊 Chart Builder")

# Step 1: Choose chart type
chart_type = st.selectbox(
    "Choose a chart type",
    ["Scatter Plot", "Histogram", "Bar Chart", "Word Cloud"]
)

# Step 2: Based on chart type, show relevant options
if chart_type == "Scatter Plot":
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    x_col = st.selectbox("Select X-axis (numeric)", options=num_cols)
    y_col = st.selectbox("Select Y-axis (numeric)", options=num_cols)

    if x_col and y_col:
        fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
        st.plotly_chart(fig, use_container_width=True)

elif chart_type == "Histogram":
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    col = st.selectbox("Select column for histogram", options=num_cols)

    if col:
        fig = px.histogram(df, x=col, nbins=20, title=f"Histogram of {col}")
        st.plotly_chart(fig, use_container_width=True)

elif chart_type == "Bar Chart":
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    col = st.selectbox("Select categorical column", options=cat_cols)

    if col:
        fig = px.bar(df[col].value_counts().reset_index(),
                     x="index", y=col, title=f"Bar Chart of {col}")
        st.plotly_chart(fig, use_container_width=True)

elif chart_type == "Word Cloud":
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    col = st.selectbox("Select text column for Word Cloud", options=cat_cols)

    if col:
        text = " ".join(df[col].dropna().astype(str))
        wordcloud = WordCloud(width=800, height=400,
                              background_color="white").generate(text)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
