import streamlit as st
import plotly.express as px
from session_state import get_data, initialize_session_state

def plot_histogram(data, col_name):
    st.write(f"### Column Name: {col_name}")
    st.write(f"#### Histogram:")
    st.info("Histogram: These plots are often used to visualize data distributions, or how values are spread out across a range. For example, if you were to create a histogram of the ages of people in a group, you would see how many people are in each age range. If most people are between 20 and 30 years old, you would see a tall bar in that range, while shorter bars would represent the fewer number of people in other age ranges.")
    fig = px.histogram(data[col_name], nbins=20, marginal="rug", opacity=0.9)
    fig.update_layout(showlegend=False, bargap=0.1)
    fig.update_layout(title="Histogram of " + col_name)  # Add plot title
    fig.update_xaxes(title_font=dict(color='black'), tickfont=dict(color='black'))  # Set x-axis label and tick color
    fig.update_yaxes(title_font=dict(color='black'), tickfont=dict(color='black'))  # Set y-axis label and tick color
    st.plotly_chart(fig, use_container_width=True)
    st.info("The x-axis represents the values in the column, while the y-axis shows the frequency or count of those values in the histogram.")

def plot_box(data, col_name):
    st.write(f"#### Box Plot:")
    st.info("Box Plot: This plot shows the distribution of the data and also helps to identify outliers if present. It displays the minimum, first quartile, median, third quartile, and maximum values of the data in a visual format. The box represents the interquartile range (IQR), while the whiskers extend to the minimum and maximum values that are not considered outliers. Any data points outside of the whiskers are considered outliers and are plotted as individual points")
    fig = px.box(data[col_name])
    fig.update_layout(title="Box Plot of " + col_name)  # Add plot title
    fig.update_xaxes(title_font=dict(color='black'), tickfont=dict(color='black'))  # Set x-axis label and tick color
    fig.update_yaxes(title_font=dict(color='black'), tickfont=dict(color='black'))  # Set y-axis label and tick color
    st.plotly_chart(fig, use_container_width=True)
    # st.info() about the inter quartile range and explaining box plot to understand

def main():
    st.set_page_config(page_title="Plots for Columns - Diagnostic Analysis Tool", page_icon=":bar_chart:", layout="wide")
    st.write("# Plots for Columns")

    # Initialize the session state
    initialize_session_state()

    # Access the stored data in other files
    data = get_data()

    if data is not None:
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        col_name = st.selectbox("Select a column to display the graph", numeric_cols)
        if data[col_name].dtype in ["float64", "int64"]:
            plot_histogram(data, col_name)
            plot_box(data, col_name)
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

if __name__ == "__main__":
    main()
