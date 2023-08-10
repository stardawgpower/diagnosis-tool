import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_regression
from session_state import get_data, initialize_session_state

# Function to format the table with CSS styling
def format_table(table):
    styles = [{'selector': 'th', 'props': [('color', 'black')]}]
    return table.style.set_table_styles(styles)

def perform_regression(data, target_col, independent_cols):
    # Imputation method
    for i in data.columns:
        data[i] = data[i].replace('[^a-zA-Z0-9 ]+', np.nan, regex=True)
        if data[i].dtype in ['int64', 'float64']:
            # Standardize numeric columns except for datetime columns
            if data[i].dtype != 'datetime64[ns]':
                scaler = StandardScaler()
                data[i] = scaler.fit_transform(data[i].values.reshape(-1, 1))
            data[i].fillna(data[i].mean(), inplace=True)
        elif data[i].dtype in ['objesct']:
            data[i].fillna(data[i].mode()[0], inplace=True)

    data.drop_duplicates(inplace=True) 

    # Assuming independent_cols and target_col have been defined earlier
    x = data[independent_cols]
    y = pd.to_numeric(data[target_col], errors='coerce').fillna(0)

    # Capping for numeric columns 
    for i in x.select_dtypes(include=['int64', 'float64']):
        percentiles = x[i].quantile([0.01, 0.99])
        x[i] = x[i].clip(percentiles.loc[0.01], percentiles.loc[0.99])

        # Standardize numeric columns except for datetime columns
        if x[i].dtype != 'datetime64[ns]':
            scaler = StandardScaler()
            x[i] = scaler.fit_transform(x[i].values.reshape(-1, 1))

    # labelEncoding
    le=LabelEncoder()
    for i in x.select_dtypes(include=['object']):
        x[i] = le.fit_transform(x[i])

    # Fit linear regression model and obtain results summary
    x_with_intercept = sm.add_constant(x)
    model = sm.OLS(y, x_with_intercept)
    results = model.fit()
    coef_df = pd.DataFrame({
        'Variable': x_with_intercept.columns,
        'Coefficient': results.params,
        'VIF': [variance_inflation_factor(x_with_intercept.values, i) for i in range(x_with_intercept.shape[1])]
    })

    coef_df = coef_df.iloc[1:]  # Remove intercept row
    coef_df['abs_coeff'] = abs(coef_df['Coefficient'])
    sorted_df = coef_df.sort_values('abs_coeff', ascending=True)
    
    # Create an interactive bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sorted_df['VIF'],
        y=sorted_df['Variable'],
        orientation='h',
        marker=dict(color='teal'),
        hovertemplate='%{y}: %{x:.3f}<extra></extra>'
    ))
    fig.update_layout(
        title='Variance Inflation Factors (VIF) of Independent Variables',
        xaxis_title='VIF Value',
        yaxis_title='Independent Variables'
    )
    fig.update_xaxes(title_font=dict(color='black'), tickfont=dict(color='black'))  # Set x-axis label and tick color
    fig.update_yaxes(title_font=dict(color='black'), tickfont=dict(color='black'))  # Set y-axis label and tick color
    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)
    st.info("VIF checks if similar data is used to predict in a regression model. Too much similarity makes it hard to predict accurately. Lower VIF means less similarity, leading to better predictions.")
    st.info("VIF helps us understand if there are any variables in our data that are closely related to each other, and that might be influencing our results in a way that we don't want that varible.")
    st.write("- In this graph, each variable is represented by a bar, and the height of the bar shows its VIF value. The bars are sorted from highest to lowest VIF, so we can quickly see which variables are more problematic.")
    st.write("- Our goal is to have all variables with a low VIF, ideally close to 1. This means that each variable provides unique information, and we can trust its contribution. If we have variables with a high VIF, we may need to remove some of them or find a way to combine them to avoid repetitiveness.")

    # Calculate feature importances
    f_values, p_values = f_regression(x, y)
    feat_importances = pd.Series(f_values, index=x.columns).sort_values()
    # Create interactive bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(x=feat_importances.values,
                        y=feat_importances.index,
                        orientation='h',
                        marker=dict(color='teal'),
                        hovertemplate='%{y}: %{x:.3f}<extra></extra>'))
    fig.update_layout(title='Feature Importances',
                    xaxis_title='F-value',
                    yaxis_title='Feature')
    fig.update_xaxes(title_font=dict(color='black'), tickfont=dict(color='black'))  # Set x-axis label and tick color
    fig.update_yaxes(title_font=dict(color='black'), tickfont=dict(color='black'))  # Set y-axis label and tick color
    st.plotly_chart(fig, use_container_width=True)
    st.info("Feature Importance scores indicate how much each factor affects a model's accuracy, but high scores don't always mean better results. Some factors may not add valuable information to the model, or they may introduce errors, even if they score high on the importance scale.")
    st.write("- The plot shows a bar chart with features listed on the y-axis and their corresponding F-values on the x-axis.")
    st.write("- This chart can help you identify which columns are the most important for predicting the target variable in your dataset.")
    st.write("- The F-score tells us how important a feature is in predicting the target variable, with a higher F-score indicating a more important feature.")
    st.write("- The bars are sorted from least to most important, so you can easily see which features have the biggest impact on the target variable.")

    # Create interactive bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(x=sorted_df['Coefficient'],
                        y=sorted_df['Variable'],
                        orientation='h',
                        marker=dict(color='teal'),
                        hovertemplate='%{y}: %{x:.3f}<extra></extra>'))
    fig.update_layout(title='Regression Coefficients of Independent Variables',
                    xaxis_title='Regression Coefficient',
                    yaxis_title='Independent Variables')
    fig.update_xaxes(title_font=dict(color='black'), tickfont=dict(color='black'))  # Set x-axis label and tick color
    fig.update_yaxes(title_font=dict(color='black'), tickfont=dict(color='black'))  # Set y-axis label and tick color
    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)
    st.info("The feature regression coefficient tells us how much each factor affects the prediction outcome. It helps us understand which factors are more important.")
    st.write("- In this case, the bars represent different independent variables, or factors, that can influence the dependent variable, or outcome, that we are interested in predicting.")
    st.write("- The taller the bar, the more important the factor is in predicting the outcome.")
    st.write("- The factors are listed from top to bottom, with the most important one at the top.")
    
    # Calculate R-squared value
    poly_features = PolynomialFeatures(degree=4) # You can change the degree as needed
    x_poly = poly_features.fit_transform(x)
    model = LinearRegression()
    model.fit(x_poly, y)
    y_pred = model.predict(x_poly)
    r2 = r2_score(y, y_pred)

    # Display table of results
    st.write('##### Regression Coefficients and VIF values:')
    st.table(format_table(coef_df[['Coefficient', 'VIF', 'abs_coeff']]))
    st.info("R-squared is a statistical measure used to evaluate how well a model fits the data. It ranges from 0 to 1, where 1 represents a perfect fit and 0 represents no fit at all. A higher score indicates better predictions.")
    st.info(f"R-squared: {r2}")

st.set_page_config(page_title="Regression Analysis - Diagnostic Analysis Tool", page_icon=":bar_chart:", layout="wide")
st.write("# Regression")

# Initialize the session state
initialize_session_state()
# Access the stored data in other files
data = get_data()

if data is not None:
    # Ask user to select target and independent variable
    all_cols = data.columns
    num_cols = data.select_dtypes(include=["float64","int64"]).columns
    date_cols = data.select_dtypes(include=["datetime64"]).columns.tolist()
    st.info("Target variable is what we want to predict. It's the outcome or result we're trying to understand from our data")
    target_col = st.selectbox("Select target variable column", num_cols)
    if set(date_cols).issubset(set(all_cols)):
        independent_cols = st.multiselect("Select independent variable columns", [col for col in all_cols if col != target_col and col not in date_cols])
    else:
        independent_cols = st.multiselect("Select independent variable columns", [col for col in all_cols if col != target_col])
    if independent_cols:
        perform_regression(data, target_col, independent_cols)
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
        