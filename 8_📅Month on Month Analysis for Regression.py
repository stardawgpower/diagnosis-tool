import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import plotly.graph_objects as go
import plotly.io as pio
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

    # to deal with outliers we can perform capping method

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

    return coef_df["Coefficient"], coef_df["VIF"]

# Function to format the table with CSS styling
def format_table(table):
    styles = [{'selector': 'th', 'props': [('color', 'black')]}]
    return table.style.set_table_styles(styles)


st.set_page_config(page_title="Month on Month Analysis for Regression - Diagnostic Analysis Tool", page_icon=":bar_chart:", layout="wide")
st.write("# Month on Month Analysis for Regression Coefficient and VIF")

# Initialize the session state
initialize_session_state()
# Access the stored data in other files
data_copy = get_data()

if data_copy is not None:
    # Make a copy of the data to prevent modifying the original dataset
    data = data_copy.copy()
    # Get the list of datetime columns
    datetime_cols = data.select_dtypes(include=["datetime64"]).columns.tolist()
    
    if len(datetime_cols) > 0:
        # Select the datetime column
        datetime_col = st.selectbox("Select the datetime column", datetime_cols)

        if datetime_col in data.columns:
            # Remove other datetime columns
            other_datetime_cols = [col for col in datetime_cols if col != datetime_col]
            data_filtered = data.drop(other_datetime_cols, axis=1)

            # Select the years for comparison
            years = pd.DatetimeIndex(data_filtered[datetime_col]).year.unique()
            selected_years = st.multiselect("Select years to compare", years) 

            # Filter the data based on the selected years and get unique months
            data_filtered = data_filtered[pd.DatetimeIndex(data_filtered[datetime_col]).year.isin(selected_years)]

            months = pd.DatetimeIndex(data_filtered[datetime_col]).month_name().unique()
            selected_months = st.multiselect("Select months to compare", months)

            # Filter the data_filtered DataFrame based on the selected months
            data_filtered = data_filtered[pd.DatetimeIndex(data_filtered[datetime_col]).month_name().isin(selected_months)]

            # Remove columns with None values
            data_filtered = data_filtered.dropna(axis=1, how="all")
            
            # Ask user to select target and independent variable
            all_cols = data_filtered.columns
            num_cols = data_filtered.select_dtypes(include=["float64","int64"]).columns
            date_col = data_filtered.select_dtypes(include=["datetime64"]).columns
            target_col = st.selectbox("Select target variable column", num_cols)

            if set(date_col).issubset(set(all_cols)):
                independent_cols = st.multiselect("Select independent variable columns", [col for col in all_cols if col != target_col and col not in date_col])
            else:
                independent_cols = st.multiselect("Select independent variable columns", [col for col in all_cols if col != target_col])

            if independent_cols:
                if st.button("Perform Month on Month Analysis for Regression Coefficient and VIF"):
                    # Create empty lists to store the results
                    regression_coefficient_results = []
                    vif_results = []

                    for month in pd.DatetimeIndex(data_filtered[datetime_col]).month_name().unique():
                        # Filter data to only include selected features and the current month
                        data_month = data_filtered[data_filtered[datetime_col].dt.month_name() == month][[target_col] + independent_cols]
                        regression_coefficient, vif = perform_regression(data_month, target_col, independent_cols)

                        regression_coefficient_results.append(regression_coefficient)
                        vif_results.append(vif)

                    # Create DataFrame from the results
                    regression_coefficient_results = pd.DataFrame(regression_coefficient_results, index=pd.DatetimeIndex(data_filtered[datetime_col]).month_name().unique(), columns=independent_cols).T
                    vif_results = pd.DataFrame(vif_results, index=pd.DatetimeIndex(data_filtered[datetime_col]).month_name().unique(), columns=independent_cols).T

                    st.subheader("Regression Coefficients")
                    st.write("Month on month regression coefficient analysis examines the relationship between a dependent variable and independent variables over time. The table showing the strength and direction of the relationship, with a positive or negative coefficient")
                    regression_coefficient_results_table = regression_coefficient_results.style.format("{:3f}").set_table_styles([{
                        'selector': 'td',
                        'props': [('text-align', 'center'), ('vertical-align', 'middle')]
                    }]).to_html(escape=False)

                    # Display the formatted table using Markdown
                    st.markdown(regression_coefficient_results_table, unsafe_allow_html=True)

                    ranking_coefficient = regression_coefficient_results.rank(ascending=True, axis=0, method='min')
                    st.subheader("Ranking the features on Regression Coefficient values")
                    ranking_coefficient_results_table = ranking_coefficient.astype(int).style.format("{}").set_table_styles([{
                        'selector': 'td',
                        'props': [('text-align', 'center'), ('vertical-align', 'middle')]
                    }]).to_html(escape=False)

                    # Display the formatted table using Markdown
                    st.markdown(ranking_coefficient_results_table, unsafe_allow_html=True)
                    
                    fig = go.Figure()
                    color_scale = pio.templates['plotly'].layout['colorway']  # Get the default color scale

                    for i, col in enumerate(ranking_coefficient.columns):
                        fig.add_trace(go.Bar(x=list(ranking_coefficient.index),
                                            y=ranking_coefficient[col].values.flatten(),
                                            name=col,
                                            marker=dict(color=color_scale[i % len(color_scale)]),
                                            hovertemplate='%{y}<extra></extra>'))

                    # Update layout
                    fig.update_layout(
                        title='Ranking of Features based on Regression Coefficients',
                        xaxis_title='Features',
                        yaxis_title='Ranking',
                        barmode='group',
                        legend_title='Months'  # Update legend title to 'Months'
                    )

                    # Set the x-axis tick labels to be the months
                    fig.update_xaxes(ticktext=list(ranking_coefficient.index), tickvals=list(range(len(ranking_coefficient.index))))
                    fig.update_xaxes(title_font=dict(color='black'), tickfont=dict(color='black'))  # Set x-axis label and tick color
                    fig.update_yaxes(title_font=dict(color='black'), tickfont=dict(color='black'))  # Set y-axis label and tick color

                    # Display the plot in Streamlit
                    st.plotly_chart(fig, use_container_width=True)
            
                    st.subheader("Variance Inflation Factors")
                    st.write("Month-wise VIF analysis assesses the degree of correlation between factors in a dataset over time, providing a table of the results to identify important factors for analysis.")
                    st.write("If two factors in a dataset are very similar, they might provide the same information and only one is needed. The table identifies the strongest correlation between factors, helping to eliminate redundancy in analysis.")

                    vif_results = vif_results.loc[vif_results.min(axis=1).sort_values(ascending=True).index]
                    vif_results_table = vif_results.style.format("{:3f}").set_table_styles([{
                        'selector': 'td',
                        'props': [('text-align', 'center'), ('vertical-align', 'middle')]
                    }]).to_html(escape=False)

                    # Display the formatted table using Markdown
                    st.markdown(vif_results_table, unsafe_allow_html=True)

                    # Calculate ranking for each feature
                    # Calculate ranking for each feature
                    ranking = vif_results.rank(ascending=True, axis=0, method='min')

                    # Handle non-finite values by replacing them with -1
                    ranking = ranking.replace([np.inf, -np.inf, np.nan], -1)

                    # Sort the ranking DataFrame by the maximum rank for each feature
                    ranking = ranking.loc[ranking.min(axis=1).sort_values(ascending=True).index]


                    # Display ranking table
                    st.subheader("Ranking of Features")
                    st.write("The table below shows the ranking of features based on VIF values for each month:")
                    ranking_vif_results_table = ranking.astype(int).style.format("{}").set_table_styles([{
                        'selector': 'td',
                        'props': [('text-align', 'center'), ('vertical-align', 'middle')]
                    }]).to_html(escape=False)

                    # Display the formatted table using Markdown
                    st.markdown(ranking_vif_results_table, unsafe_allow_html=True)

                    fig = go.Figure()
                    color_scale = pio.templates['plotly'].layout['colorway']  # Get the default color scale

                    for i, col in enumerate(ranking.columns):
                        values = ranking[col].values.T.flatten()
                        valid_values = values[~np.isnan(values)]  # Remove NaN values
                        valid_index = [idx for idx, val in enumerate(values) if not np.isnan(val)]  # Get corresponding valid index
                        
                        fig.add_trace(go.Bar(x=[ranking.index[idx] for idx in valid_index],
                                            y=valid_values,
                                            name=col,
                                            orientation='v',
                                            marker=dict(color=color_scale[i % len(color_scale)]),
                                            hovertemplate='%{y}<extra></extra>'))

                    fig.update_layout(
                        title='Ranking of Features based on VIF Values',
                        xaxis_title = 'Features',
                        yaxis_title= 'Ranking',
                        barmode='group',
                        legend_title='months'
                    )
                    fig.update_xaxes(title_font=dict(color='black'), tickfont=dict(color='black'))  # Set x-axis label and tick color
                    fig.update_yaxes(title_font=dict(color='black'), tickfont=dict(color='black'))  # Set y-axis label and tick color

                    # Display the plot in Streamlit
                    st.plotly_chart(fig, use_container_width=True)
                   
    else:
        st.warning("No datetime column present in the dataset!")
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

