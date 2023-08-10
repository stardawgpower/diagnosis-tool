import streamlit as st
import pandas as pd
from session_state import get_data, initialize_session_state

def get_column_info(data, target_col):
    column_info = {}
    column_info["dtype"] = data[target_col].dtype

    if column_info["dtype"] in ["float64", "int64"]:
        column_info["mean"] = round(data[target_col].mean(), 2)
        column_info["median"] = round(data[target_col].median(), 2)
        column_info["mode"] = round(data[target_col].mode()[0], 2)
        column_info["missing_rows"] = data[target_col].isna().sum()
        column_info["missing_percentage"] = round(data[target_col].isna().sum() / data.shape[0] * 100, 2)
        column_info["min"] = round(data[target_col].min(), 2)
        column_info["max"] = round(data[target_col].max(), 2)
        column_info["std"] = round(data[target_col].std(), 2)
    elif column_info["dtype"] == "object":
        column_info["missing_rows"] = data[target_col].isna().sum()
        column_info["missing_percentage"] = round(data[target_col].isna().sum() / data.shape[0] * 100, 2)
        column_info["unique_values"] = data[target_col].nunique()
        column_info["value_counts"] = data[target_col].value_counts()
        column_info["value_percentages"] = (column_info["value_counts"] / len(data[target_col])) * 100
    elif column_info["dtype"] == "datetime64[ns]":
        column_info["missing_rows"] = data[target_col].isna().sum()
        column_info["missing_percentage"] = round(data[target_col].isna().sum() / data.shape[0] * 100, 2)
        column_info["oldest_record"] = data[target_col].min()
        column_info["latest_record"] = data[target_col].max()
        column_info["duration"] = data[target_col].max() - data[target_col].min()
        column_info["average_interval"] = data[target_col].diff().mean()
        # more logic can be add here according to the dataset
        column_info["missing_periods"] = data[data[target_col].diff() > pd.Timedelta(days=1)]

    return column_info

def basic_info(data):
    st.write("## Dataset Information")
    st.info(f"##### Question: What is the total number of rows present in the dataset?\n Answer: {data.shape[0]}")
    st.info(f"##### Question: What is the total number of columns or features present in the dataset?\n Answer: {data.shape[1]}")
    st.write("## Column Information")

    target_col = st.selectbox("Select a column", data.columns)

    st.write(f"##### Column Name: {target_col}")
    column_info = get_column_info(data, target_col)
    if column_info["dtype"] in ["float64", "int64"]:
        num_col1, num_col2 = st.columns(2)
        with num_col1:
            st.info(f"##### Question: What is the data type of this column?\n ##### Answer: {column_info['dtype']}")
            st.info(f"##### Question: What is the average value (Mean) of this column?\n ##### Answer: {column_info['mean']}")
            st.info(f"##### Question: What is the middle value (Median) of this column?\n ##### Answer: {column_info['median']}")
            st.info(f"##### Question: What is the most occurring value (Mode) of this column?\n ##### Answer: {column_info['mode']}")
        with num_col2:
            st.info(f"##### Question: How many rows are missing in this column?\n ##### Answer: {column_info['missing_rows']} ({column_info['missing_percentage']}%)")
            st.info(f"##### Question: What is the minimum value of this column?\n ##### Answer: {column_info['min']}")
            st.info(f"##### Question: What is the maximum value of this column?\n ##### Answer: {column_info['max']}")
            st.info(f"##### Question: How much dispersed the data is in relation to the mean (Standard Deviation)?\n ##### Answer: {column_info['std']}")
    elif column_info["dtype"] == "object":
        obj_col1, obj_col2 = st.columns(2)
        with obj_col1:
            st.info(f"##### Question: What is the data type of this column?\n ##### Answer: {column_info['dtype']}")
            st.info(f"##### Question: How many rows are missing in this column?\n ##### Answer: {column_info['missing_rows']} ({column_info['missing_percentage']}%)")
            st.info(f"##### Question: How many number of unique values present in this column?\n ##### Answer: {column_info['unique_values']}")
        with obj_col2:
            st.info("##### Question: What is the count and percentage of each unique value?\n ##### Answer: ")
            for value, count, percentage in zip(column_info["value_counts"].index, column_info["value_counts"], column_info["value_percentages"]):
                st.write(f"- Value: {value} | Count: {count} | Percentage: {percentage:.2f}%")
    elif column_info["dtype"] == "datetime64[ns]":
        dt_col1, dt_col2 = st.columns(2)
        with dt_col1:
            st.info(f"##### Question: What is the data type of this column?\n ##### Answer: {column_info['dtype']}")
            st.info(f"##### Question: What is the oldest record date of this column?\n ##### Answer: {column_info['oldest_record']}")
            st.info(f"##### Question: What is the latest record date of this column?\n ##### Answer: {column_info['latest_record']}")
        with dt_col2:
            st.info(f"##### Question: How many rows are missing in this column?\n ##### Answer: {column_info['missing_rows']} ({column_info['missing_percentage']}%)")
            st.info(f"##### Question: What is the time duration covered by the dataframe?\n ##### Answer: {column_info['duration']}")
            st.info(f"##### Question: What is the average time interval between consecutive datetimes in the dataframe?\n ##### Answer: {column_info['average_interval']}")
        if len(column_info["missing_periods"]) > 0:
            st.info("##### Question: Are there any specific time periods during which the data is missing or sparse?\n ##### Answer: Yes")
            st.write(column_info["missing_periods"])
        else:
            st.info("##### Question: Are there any specific time periods during which the data is missing or sparse?\n ##### Answer: No")

    return data

def main():
    st.set_page_config(page_title="Statistics - Diagnostic Analysis Tool", page_icon=":bar_chart:", layout="wide")
    st.write("# Statistics about the Dataset")
    st.info("This displays the critical statistics for a selected column of a given dataset, such as the mean, median, mode, standard deviation, minimum value, and maximum value. These statistics can provide insights into the distribution and characteristics of the data, such as its central tendency, variability, and outliers.")
    st.write("##### For Example:")
    st.write("- The mean represents the average value of the data points, the median represents the middle value, and the mode represents the most common value. This tells about the central tendency of the data.")
    st.write("- The standard deviation indicates how spread out the data points are from the mean, while the minimum and maximum values can show the range of values in the data.")

    initialize_session_state()
    data = get_data()

    if data is not None:
        basic_info(data)
    else:
        st.warning("No data found! Please upload a dataset.")

    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
