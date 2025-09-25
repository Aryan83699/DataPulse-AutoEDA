# DataPulse - Interactive Data Analysis App

[![Deploy on Render](https://img.shields.io/badge/Deploy-Render-blue)](https://datapulse-autoeda.onrender.com)

## 🔗 Live Demo
Access the live app here: [DataPulse on Render](https://datapulse-autoeda.onrender.com)

---

## Overview
**DataPulse** is an interactive **Streamlit** web application for automated exploratory data analysis (EDA).  
It allows users to upload CSV datasets and perform various analyses, including:

- **Data Exploration**: View dataset overview, sample rows, and summary statistics.  
- **Variable Analysis**: Analyze individual columns with detailed numeric/categorical metrics.  
- **Missing Values Handling**: Identify and impute missing data using mean, median, mode, or iterative imputer.  
- **Visualizations**: Create interactive 2D and 3D plots using Plotly, Matplotlib, and Seaborn.  
- **WordCloud** generation for textual/categorical data.  

---

## Features
- Easy CSV upload via sidebar
- Automatic type detection and column categorization
- Statistical tests (T-test, Z-test, Chi-squared)
- Multiple chart types: Line, Scatter, Bar, Box, Violin, Histogram, Pie, Bubble
- Interactive 3D plots

---

## Tech Stack
- **Frontend/Backend**: Streamlit  
- **Data Analysis**: Pandas, NumPy, SciPy, StatsModels, Scikit-learn  
- **Visualizations**: Matplotlib, Seaborn, Plotly, WordCloud  
- **Other Libraries**: streamlit-option-menu  

---

## Installation
1. Clone the repository:
```bash
git clone https://github.com/Aryan83699/DataPulse-AutoEDA.git
cd DataPulse-AutoEDA

2. Install Dependencies
pip install -r requirements.txt

3. Run the app
streamlit run app.py
