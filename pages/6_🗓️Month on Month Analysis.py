import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from session_state import get_data, initialize_session_state

def plot_line_chart(data, x_col, y_cols, title, value):
    traces = []
    # Iterate over each y column and create a Scatter trace for each one
    for y_col in y_cols:
        trace = go.Scatter(x=data[x_col], y=data[y_col], mode='lines+markers', name=y_col)
        traces.append(trace)
    # Create a layout for the chart
    layout = go.Layout(
        title=title,
        xaxis=dict(title=x_col),
        yaxis=dict(title=value),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=14, color='black'),
        margin=dict(l=100, r=100, t=80, b=80),
        width=800,
        height=600
    )
    # Create a Figure object using the traces and layout
    fig = go.Figure(data=traces, layout=layout)
    fig.update_xaxes(title_font=dict(color='black'), tickfont=dict(color='black'))  # Set x-axis label and tick color
    fig.update_yaxes(title_font=dict(color='black'), tickfont=dict(color='black'))  # Set y-axis label and tick color
    # Use Streamlit's plotly_chart function to display the chart
    st.plotly_chart(fig, use_container_width=True)

def df_trasformation(results, months):
    selected_months = months
    # Get the results DataFrame for the selected months and numerical features
    results = results[selected_months]
    # Transpose the DataFrame so that each row represents a feature and each column represents a month
    results = results.T.reset_index()
    # Rename the index column to "Month"
    results = results.rename(columns={"index": "Month"})
    return results

# Function to format the table with CSS styling
def format_table(table):
    styles = [{'selector': 'th', 'props': [('color', 'black')]}]
    return table.style.set_table_styles(styles)

def mom(data_filtered, selected_months, datetime_col):
    num_cols = data_filtered.select_dtypes(include=["float64", "int64"]).columns
    obj_cols = data_filtered.select_dtypes(include="object").columns

    # Numeric data type comparison
    if not num_cols.empty:
        # Select features for comparison
        select_feature = st.multiselect('Select numerical features to compare', num_cols)
        if select_feature:
            # Create empty dataframes to store the results for each summary statistic
            min_results = pd.DataFrame(index=select_feature)  
            max_results = pd.DataFrame(index=select_feature) 
            sum_results = pd.DataFrame(index=select_feature)
            mean_results = pd.DataFrame(index=select_feature)
            median_results = pd.DataFrame(index=select_feature)
            mode_results = pd.DataFrame(index=select_feature)
            missing_results = pd.DataFrame(index=select_feature)
            skewness_results = pd.DataFrame(index=select_feature)
            kurtosis_results = pd.DataFrame(index=select_feature)
            outliers_results = pd.DataFrame(index=select_feature)
            std_results = pd.DataFrame(index=select_feature)
        
            # Loop over the selected months and calculate summary statistics for each month
            for month in selected_months:
                # Filter data to only include selected features and the current month
                data_filtered = data[data[datetime_col].dt.month_name() == month][select_feature]
                # Calculate summary statistics for the current month
                minimum = data_filtered.min() 
                maximum = data_filtered.max() 
                total_sum = data_filtered.sum()
                mean = data_filtered.mean()
                median = data_filtered.median()
                mode = data_filtered.mode().iloc[0]
                missing = data_filtered.isna().sum()
                skewness = data_filtered.skew()
                kurtosis = data_filtered.kurtosis()
                outliers = ((data_filtered < (mean - 3 * data_filtered.std())) | (data_filtered > (mean + 3 * data_filtered.std()))).sum()
                std = data_filtered.std()
                
                # Add the results for the current month to the corresponding dataframes
                mean_results[month] = mean
                median_results[month] = median
                mode_results[month] = mode
                missing_results[month] = missing
                skewness_results[month] = skewness
                kurtosis_results[month] = kurtosis
                outliers_results[month] = outliers
                std_results[month] = std
                min_results[month] = minimum  
                max_results[month] = maximum  
                sum_results[month] = total_sum
            
            # Display the results for each summary statistic as a separate table
            st.write("### Numeric Data Type Comparison")

            st.write("#### Minimum Value")
            st.table(format_table(min_results))
            min_results = df_trasformation(min_results, selected_months)
            plot_line_chart(min_results, "Month", select_feature, "Minimum Comparison for Selected Months and Features", "Minimum Value")

            st.write("#### Maximum Value")
            st.table(format_table(max_results))
            max_results = df_trasformation(max_results, selected_months)
            plot_line_chart(max_results, "Month", select_feature, "Maximum Comparison for Selected Months and Features", "Maximum Value")

            st.write("#### Total Sum")
            st.table(format_table(sum_results))
            sum_results = df_trasformation(sum_results, selected_months)
            plot_line_chart(sum_results, "Month", select_feature, "Total Sum Comparison for Selected Months and Features", "Total Sum")

            st.write("#### Mean")
            st.table(format_table(mean_results))
            mean_results = df_trasformation(mean_results, selected_months) 
            plot_line_chart(mean_results, "Month", select_feature, "Mean Comparison for Selected Months and Features", "Mean")

            st.write("#### Median")
            st.table(format_table(median_results))
            median_results = df_trasformation(median_results, selected_months)
            plot_line_chart(median_results, "Month", select_feature, "Median Comparison for Selected Months and Features", "Median")

            st.write("#### Mode")
            st.table(format_table(mode_results))
            mode_results = df_trasformation(mode_results, selected_months)
            plot_line_chart(mode_results, "Month", select_feature, "Mode Comparison for Selected Months and Features", "Mode")

            st.write("#### Missing Values")
            st.table(format_table(missing_results))
            missing_results = df_trasformation(missing_results, selected_months)
            plot_line_chart(missing_results, "Month", select_feature, "Missing Values Comparison for Selected Months and Features", "Missing Values")

            st.write("#### Skewness")
            st.info("Skewness tells us whether the data are more spread out on one side of the average value than the other. If there are more values on the right side of the average, it has negative skewness. If there are more values on the left side of the average, it has positive skewness.")
            st.table(format_table(skewness_results))
            st.info("Positive skewness: A positive skewness value indicates that the distribution is skewed to the right, with the tail of the distribution longer on the right side. It means more values are on the left side of the mean. For example, a positive skewness in an income dataset would mean more people have lower incomes than the average income.")
            st.info("Negative skewness: A negative skewness value indicates that the distribution is skewed to the left, with the tail of the distribution longer on the left side. It means more values are on the mean's right side. For example, a negative skewness in a dataset of test scores would mean that more students scored higher than the average score.")
            skewness_results = df_trasformation(skewness_results, selected_months)
            plot_line_chart(skewness_results, "Month", select_feature, "Skewness Comparison for Selected Months and Features", "Skewness")

            st.write("#### Kurtosis")
            st.info("Kurtosis tells us how 'peaked' or 'flat' the data distribution is, compared to a normal distribution. A normal distribution is symmetrical and has a bell-shaped curve. A distribution with a high kurtosis has a more peaked shape than a normal distribution, which means it has more extreme values or outliers. A distribution with a low kurtosis is flatter and broader than a normal distribution, which means it has fewer extreme values or outliers.")
            st.table(format_table(kurtosis_results))
            st.info("Positive kurtosis: A positive kurtosis value indicates that the distribution has sharper peaks and thicker tails than a normal distribution. It means the data have more extreme values or outliers than a normal distribution. For example, a positive kurtosis in a dataset of stock prices would mean that there are more stocks with extremely high or low prices than in a normal distribution.")
            st.info("Negative kurtosis: A negative kurtosis value indicates that the distribution has flatter peaks and thinner tails than a normal distribution. It means the data have fewer extreme values or outliers than a normal distribution. For example, a negative kurtosis in a dataset of the apples' weight would mean fewer apples with extremely high or low weights than a normal distribution.")
            kurtosis_results = df_trasformation(kurtosis_results, selected_months)
            plot_line_chart(kurtosis_results, "Month", select_feature, "Kurtosis Comparison for Selected Months and Features", "kurtosis")

            st.info("Example: If a distribution has a positive skewness, it suggests that the mean may not represent the typical value well, and the median may be a better measure of central tendency. Similarly, a high kurtosis indicates that the data may have more extreme values or outliers, which could impact the analysis.")
            st.write("#### Outliers")
            st.table(format_table(outliers_results))
            outliers_results = df_trasformation(outliers_results, selected_months)
            plot_line_chart(outliers_results, "Month", select_feature, "Outliers Comparison for Selected Months and Features", "Outliers")

            st.write("#### Standard Deviation")
            st.info("Standard deviation is a statistical measure that tells us how far the data is from the average value. It is a way to understand the variability or dispersion of the data in a dataset.")
            st.table(std_results)
            st.info("The standard deviation helps us understand how much the individual data points in a dataset vary from the average value. A small standard deviation means the data points are clustered closely around the average value. In contrast, a large standard deviation indicates that the data points are more spread out from the average value.")
            std_results = df_trasformation(std_results, selected_months)
            plot_line_chart(std_results, "Month", select_feature, "Standard Deviation Comparison for Selected Months and Features", "Standard Deviation")

    # Object data type comparison
    if not obj_cols.empty:
        # Select features for comparison
        select_feature = st.multiselect('Select object features to compare', obj_cols)

        if select_feature:
            # Create empty dataframes to store the results for each summary statistic
            unique_results = pd.DataFrame(index=select_feature)
            missing_results = pd.DataFrame(index=select_feature)

            # Loop over the selected months and calculate summary statistics for each month
            for month in selected_months:
                # Filter data to only include selected features and the current month
                data_filtered = data[data[datetime_col].dt.month_name() == month][select_feature]

                # Calculate summary statistics for the current month
                unique = data_filtered.nunique()
                missing = data_filtered.isna().sum()

                # Add the results for the current month to the corresponding dataframes
                unique_results[month] = unique
                missing_results[month] = missing
            
            # Display the results for each summary statistic as a separate table
            st.write("### Object Data Type Comparison")
            st.write("#### Unique Values")
            st.table(format_table(unique_results))
            st.write("#### Missing Values")
            st.table(format_table(missing_results))

st.set_page_config(page_title="Month on Month Analysis - Diagnostic Analysis Tool", page_icon=":bar_chart:", layout="wide")
st.write("# Month on Month Analysis")
st.info("In this analysis, you can quickly see how different factors change with time. It can help you discover patterns and trends in your data. You can choose the date-time column for which you want to do the analysis. Based on that, you must select a year or multiple years, months, and features you want to include in the analysis.")

# Initialize the session state
initialize_session_state()
# Access the stored data in other files
data = get_data()

if data is not None:
    # Get the list of datetime columns
    datetime_cols = data.select_dtypes(include=["datetime64"]).columns.tolist()
    
    if len(datetime_cols) > 0:
        # Select the datetime column
        datetime_col = st.selectbox("Select the datetime column", datetime_cols)

        if datetime_col in data.columns:
            # Select the years for comparison
            years = pd.DatetimeIndex(data[datetime_col]).year.unique()
            selected_years = st.multiselect("Select years to compare", years) 
            # Filter the data based on the selected years and get unique months
            data_filtered = data[pd.DatetimeIndex(data[datetime_col]).year.isin(selected_years)]
            # Remove columns with None values
            data_filtered = data_filtered.dropna(axis=1, how="all")
            # Replace missing values with the mode in object columns
            object_cols = data_filtered.select_dtypes(include=["object"]).columns
            if not object_cols.empty:
                data_filtered[object_cols] = data_filtered[object_cols].fillna(data_filtered[object_cols].mode().iloc[0])

            if datetime_col in data_filtered.columns:
                months = pd.DatetimeIndex(data_filtered[datetime_col]).month_name().unique()
                selected_months = st.multiselect("Select months to compare", months)
                mom(data_filtered, selected_months, datetime_col)
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