import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import plotly.graph_objs as go
from session_state import get_data, initialize_session_state

def preprocess_data(data):
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

    categorical_cols = data.select_dtypes(include=['object']).columns
    data[categorical_cols] = data[categorical_cols].fillna('Unknown')

    scaler = StandardScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

    le = LabelEncoder()
    data[categorical_cols] = data[categorical_cols].apply(le.fit_transform)

    return data

def perform_regression(data, target_col, independent_cols):
    data = preprocess_data(data)

    x = data[independent_cols]
    y = pd.to_numeric(data[target_col], errors='coerce').fillna(0)

    x_with_intercept = sm.add_constant(x)
    model = sm.OLS(y, x_with_intercept)
    results = model.fit()

    coef_df = pd.DataFrame({
        'Variable': x_with_intercept.columns[1:],
        'Coefficient': results.params[1:],
        'VIF': [variance_inflation_factor(x_with_intercept.values, i) for i in range(1, x_with_intercept.shape[1])]
    })
    coef_df['abs_coeff'] = abs(coef_df['Coefficient'])

    return coef_df, results, x_with_intercept, results.params

def what_if(coef_df, independent_cols, target_col, data, exog, params):
    # To filter independent_cols to only select numeric columns for what-if
    what_if_feature = st.selectbox('Select feature to perform what-if analysis', independent_cols)
    what_if_percentage = st.select_slider('Use the scrollbar to select the percentage change', options=['1%','2%','3%','4%','5%','6%','7%','8%','9%','10%','11%','12%','13%','14%','15%','16%','17%','18%','19%','20%','21%','22%','23%','24%','25%'])

    coef = coef_df.loc[what_if_feature, 'Coefficient']
    percentage_change = int(what_if_percentage.strip('%')) / 100
    target_change = coef * percentage_change * 100

    x = data[independent_cols].copy()
    x[what_if_feature] = x[what_if_feature] * (1 + percentage_change)
    x = preprocess_data(x)
    exog = sm.add_constant(x)
    y_pred = np.dot(exog, params)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data[what_if_feature], y=data[target_col], mode='markers', name='Original Data'))
    fig.add_trace(go.Scatter(x=x[what_if_feature], y=y_pred, mode='markers', name='What-if Analysis'))
    fig.update_layout(title=f"What-if Analysis for {what_if_feature} with {what_if_percentage} increase", xaxis_title=what_if_feature, yaxis_title=target_col)
    fig.update_xaxes(title_font=dict(color='black'), tickfont=dict(color='black'))  # Set x-axis label and tick color
    fig.update_yaxes(title_font=dict(color='black'), tickfont=dict(color='black'))  # Set y-axis label and tick color
    st.plotly_chart(fig, use_container_width=True)

    st.info(f"A {what_if_percentage} increase in {what_if_feature} is predicted to result in a {target_change:.2f}% increase in the target variable.")

st.set_page_config(page_title="What-If Analysis - Diagnostic Analysis Tool", page_icon=":bar_chart:", layout="wide")
st.write("# What-If Analysis")
st.info("What-if analysis is to help you make informed decisions by understanding the relationships between different factors. It allows you to explore different scenarios and see how changes in one variable affect the outcome or prediction. This analysis helps you understand the potential consequences of those changes and make better choices based on the insights gained.")
st.write("##### For Example:")
st.write("- Let's say you have a machine that predicts the price of a house based on factors like its size, number of rooms, and location. Now, you want to know how changing one of these factors would affect the predicted price.")
st.write("- A what-if analysis helps you understand the impact of changing one of those factors on the predicted price. For example, let's say you want to know what would happen if you added an extra room to a house. You would select the number of rooms as the variable to analyse.")
st.write("- Next, you define the range of changes you want to explore. For instance, see the effect of adding anywhere from 1 to 3 rooms to the house.")
st.write("- You might change(increase/decrease) the number of rooms by any no. Let us say one and observe how the predicted price changes based on that modification.")
st.write("- By doing this, you can understand how changes in the number of rooms impact the predicted price. Adding an extra room increases the expected price, indicating that having more rooms makes the house more valuable.")

initialize_session_state()

data = get_data()

# make data copy to preserve the original dataset

if data is not None:
    num_cols = data.select_dtypes(include=["float64", "int64"]).columns
    date_cols = data.select_dtypes(include=["datetime64"]).columns.tolist()
    target_col = st.selectbox("Select target variable column", num_cols)
    if set(date_cols).issubset(set(data.columns)):
        independent_cols = st.multiselect("Select independent variable columns", [col for col in data.columns if col != target_col and col not in date_cols])
    else:
        independent_cols = st.multiselect("Select independent variable columns", [col for col in data.columns if col != target_col])
    if independent_cols:
        coef_df, model, exog, params = perform_regression(data, target_col, independent_cols)
        what_if(coef_df, independent_cols, target_col, data, exog, params)
else:
    st.warning("No data found! Please upload a dataset.")

# Hide Streamlit footer note
hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
