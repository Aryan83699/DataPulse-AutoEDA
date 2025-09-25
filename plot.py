import plotly.express as px
import streamlit as st
import pandas as pd 

# =================== CSS STYLING ===================
st.markdown("""
    <style>
    /* General app background */
    .stApp {
        background-color: #f8f9fa;
        font-family: "Segoe UI", Tahoma, sans-serif;
    }

    /* Titles */
    h1, h2, h3 {
        color: #2c3e50;
        text-align: center;
        font-weight: 600;
    }

    /* Selectboxes and widgets */
    .stSelectbox label {
        font-weight: 600;
        color: #34495e;
    }

    /* Chart title section */
    .css-1aumxhk {
        text-align: center;
    }

    /* Padding for plots */
    .plot-container {
        padding: 12px;
        border-radius: 12px;
        background: #ffffff;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# =================== PLOTTING FUNCTION ===================
def ploting(df):
    st.markdown("## 📊 Data Visualization Playground")
    type = st.selectbox("🔽 Select Type of Plot", options=['2D Plots','3D Plots'])
    
    if type == '2D Plots':
        plot = st.selectbox("🎨 Choose the type of chart", 
                            options=['None','Bar Plot','Box Plot','Violin Plot',
                                     'Histogram','Line Plot','Scatter Plot','Pie Chart','Bubble Chart'])

        numeric_column = df.select_dtypes(include=['number']).columns.tolist()
        cat_column = df.select_dtypes(include=['number','category']).columns.tolist()

        if plot == 'Line Plot':
            temp_x = st.selectbox("📌 Select X Axis", options=[None]+cat_column)
            temp_y = st.selectbox("📌 Select Y Axis", options=[None]+numeric_column)
            fig = px.line(df, x=temp_x, y=temp_y, title=f"📈 {temp_x} vs {temp_y} Line Plot")
            st.plotly_chart(fig)

        elif plot == 'Scatter Plot':
            temp_x1 = st.selectbox("📌 Select X Axis", options=[None]+numeric_column)
            temp_y1 = st.selectbox("📌 Select Y Axis", options=[None]+numeric_column)
            fig1 = px.scatter(df, x=temp_x1, y=temp_y1, title=f"⚡ {temp_x1} vs {temp_y1} Scatter Plot")
            st.plotly_chart(fig1)

        elif plot == 'Box Plot':
            y_axis = st.selectbox("📦 Select Y Axis (Numeric)", options=numeric_column)
            x_axis = st.selectbox("📦 Optional X Axis (Categorical)", options=[None] + cat_column)
            if x_axis:
                fig = px.box(df, x=x_axis, y=y_axis, title=f"📦 {y_axis} distribution by {x_axis}")
            else:
                fig = px.box(df, y=y_axis, title=f"📦 {y_axis} distribution")
            st.plotly_chart(fig)

        elif plot == 'Violin Plot':
            y_axis2 = st.selectbox("🎻 Select Y Axis", options=numeric_column)
            x_axis2 = st.selectbox("🎻 Select X Axis", options=[None]+cat_column)
            if x_axis2:
                fig = px.violin(df, x=x_axis2, y=y_axis2, title=f"🎻 {x_axis2} vs {y_axis2} Violin Plot")
            else:
                fig = px.violin(df, y=y_axis2, title=f"🎻 {y_axis2} Distribution")
            st.plotly_chart(fig)

        elif plot == 'Histogram':
            x_axis3 = st.selectbox("📊 Select Numeric Column", options=numeric_column)
            fig3 = px.histogram(df, x=x_axis3, title=f"📊 {x_axis3} Distribution")
            st.plotly_chart(fig3)

        elif plot == 'Pie Chart':
            x_axis4 = st.selectbox("🥧 Select Names Column", options=cat_column)
            y_axis4 = st.selectbox("🥧 Select Values Column", options=numeric_column)
            fig4 = px.pie(df, names=x_axis4, values=y_axis4, title=f"🥧 {x_axis4} and {y_axis4} Pie Chart")
            st.plotly_chart(fig4)

        elif plot == 'Bubble Chart':
            x_axis5 = st.selectbox("🔵 Select X Axis", options=numeric_column)
            y_axis5 = st.selectbox("🔵 Select Y Axis", options=numeric_column)
            size_axis = st.selectbox("🔵 Select Size Axis", options=numeric_column)
            fig5 = px.scatter(df, x=x_axis5, y=y_axis5, size=size_axis, 
                              title=f"🔵 {x_axis5} vs {y_axis5} Bubble Chart")
            st.plotly_chart(fig5)

        elif plot == 'Bar Plot':
            x_axis6 = st.selectbox("📊 Select X Axis", options=cat_column)
            y_axis6 = st.selectbox("📊 Select Y Axis", options=numeric_column)
            fig6 = px.bar(df, x=x_axis6, y=y_axis6, title=f"📊 {x_axis6} vs {y_axis6} Bar Plot")
            st.plotly_chart(fig6)

    elif type == '3D Plots':
        plot3d = st.selectbox("🧊 Choose the type of 3D Chart",
                              options=[None,'3D Scatter Plot','3D Line Plot','3D Bubble Plot'])
        numeric_column = df.select_dtypes(include=['number']).columns.tolist()
        cat_column = df.select_dtypes(include=['number','category']).columns.tolist()

        if plot3d == '3D Scatter Plot':
            x13d = st.selectbox("📌 Select X Axis", options=[None]+numeric_column)
            y13d = st.selectbox("📌 Select Y Axis", options=[None]+numeric_column)
            z13d = st.selectbox("📌 Select Z Axis", options=[None]+numeric_column)
            fig13d = px.scatter_3d(df, x=x13d, y=y13d, z=z13d, title="🧊 3D Scatter Plot", height=800, width=600)
            st.plotly_chart(fig13d)

        elif plot3d == '3D Line Plot':
            x23d = st.selectbox("📌 Select X axis", options=[None]+cat_column)
            y23d = st.selectbox("📌 Select Y axis", options=[None]+numeric_column)
            z23d = st.selectbox("📌 Select Z axis", options=[None]+numeric_column)
            fig23d = px.line_3d(df, x=x23d, y=y23d, z=z23d, title="🧊 3D Line Plot", height=800, width=600)
            st.plotly_chart(fig23d)

        # elif plot3d == '3D Surface Plot':
        #     x33d = st.selectbox("📌 Select X Axis", options=numeric_column)
        #     y33d = st.selectbox("📌 Select Y Axis", options=numeric_column)
        #     z33d = st.selectbox("📌 Select Z Axis", options=numeric_column)
        #     temp_df = df.copy()
        #     temp_df['xbins'] = pd.cut(temp_df[x33d], bins=20)
        #     temp_df['ybins'] = pd.cut(temp_df[y33d], bins=20)
        #     z = df.pivot_table(index='ybins', columns='xbins', values=z33d, aggfunc='mean')
        #     Z = z.values
        #     fig33d = px.imshow(Z, title="🧊 3D Surface Plot")
        #     st.plotly_chart(fig33d)

        elif plot3d == '3D Bubble Plot':
            x23d = st.selectbox("📌 Select X Axis", options=numeric_column)
            y23d = st.selectbox("📌 Select Y Axis", options=numeric_column)
            z23d = st.selectbox("📌 Select Z Axis", options=numeric_column)
            Size = st.selectbox("📌 Select Size Factor", options=[None]+numeric_column)
            fig23d = px.scatter_3d(df, 
                                   x=df[x23d].fillna(df[x23d].mean()), 
                                   y=df[y23d].fillna(df[y23d].mean()), 
                                   z=df[z23d].fillna(df[z23d].mean()), 
                                   size=Size, title="🧊 3D Bubble Plot", height=800, width=600)
            st.plotly_chart(fig23d)
