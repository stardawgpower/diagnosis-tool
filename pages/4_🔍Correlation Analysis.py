import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from session_state import get_data, initialize_session_state

def correlation(data):
    corr = data.corr()
    fig = px.imshow(corr, color_continuous_scale="Viridis")
    title="Correlation Matrix"
    # Add annotations to display the correlation value without hovering
    annotations = [
        go.layout.Annotation(
            x=i, y=j,
            text=str(round(corr.iloc[i, j], 2)),
            font=dict(color="black", size=14),
            showarrow=False,
            xref="x", yref="y"
        )
        for i in range(len(corr.index))
        for j in range(len(corr.columns))
    ]
    
    fig.update_layout(
        title=title,
        title_x=0.37,
        width=800, height=800, showlegend=False,
        annotations=annotations,
        coloraxis_colorbar=dict(title="Correlation", titleside="right", thickness=15, len=0.75)
    )
    fig.update_xaxes(title_font=dict(color='black'), tickfont=dict(color='black'))  # Set x-axis label and tick color
    fig.update_yaxes(title_font=dict(color='black'), tickfont=dict(color='black'))  # Set y-axis label and tick color
    fig.update_traces(hovertemplate="Feature 1: %{x}<br>Feature 2: %{y}<br>Correlation: %{z}<extra></extra>")
    st.plotly_chart(fig, use_container_width=True)
    st.info("A correlation matrix is a table that shows how all variables in a dataset are related using correlation coefficients. This helps identify meaningful relationships and is used in various fields for decision-making. Correlation coefficient values can range from -1 to 1.")
    st.info("A value close to 1 means a strong positive relationship between the variables, while a value close to -1 indicates a strong negative relationship. When the correlation value is close to 0, there is no relationship between the two variables.")

    # Select data type
    num_cols = data.select_dtypes(include=["float64", "int64"]).columns

    # Column selection
    target_col = st.selectbox("Select target variable column (the variable we want to predict)", num_cols)

    # Get correlation values with target variable
    corr_with_target = corr[target_col].sort_values(ascending=False)

    # Create a trace for the line plot
    trace = go.Scatter(
        x=corr_with_target.index,
        y=corr_with_target.values,
        mode='lines+markers',
        marker=dict(size=10, color='black'),
        line=dict(width=2, color='black')
    )

    # Create the layout
    layout = go.Layout(
        title=f'Correlation with {target_col}',
        title_x=0.3,
        xaxis=dict(title='Features', tickangle=45, color='lightgray'),
        yaxis=dict(title='Correlation', color='lightgray'),
        margin=dict(l=100, r=100, t=80, b=80),  # Adjust margins
        width=800, height=600,  # Set width and height
        plot_bgcolor='white',  # Set plot background color
        paper_bgcolor='white',  # Set paper background color
        font=dict(size=14, color='black')  # Set font size and color
    )

    # Create the figure and plot it
    fig = go.Figure(data=[trace], layout=layout)

    # Center the plot and add padding
    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=50, r=50, t=50, b=50),
        autosize=True
    )
    fig.update_xaxes(title_font=dict(color='black'), tickfont=dict(color='black'))  # Set x-axis label and tick color
    fig.update_yaxes(title_font=dict(color='black'), tickfont=dict(color='black'))  # Set y-axis label and tick color
    fig.update_traces(hovertemplate="Feature: %{x}<br>Correlation: %{y}<extra></extra>")
    st.plotly_chart(fig, use_container_width=True)
    st.info("""
    - This plot helps us understand how different features in our dataset are related to the target variable.\n
    - The x-axis shows the names of the different features, and the y-axis shows the correlation value.\n
    - This plot shows us a line for each feature, with the line height representing the correlation value.
    """)

def main():
    st.set_page_config(page_title="Correlation Analysis - Diagnostic Analysis Tool", page_icon=":bar_chart:", layout="wide")
    st.write("# Correlation Analysis")
    st.info("Correlation analysis helps us to understand the relationship between two variables. This is useful in many business, medicine, and social science areas.")
    st.write("##### For Example:")
    st.write("If we study the relationship between exercise and good health, we can use correlation analysis to determine how closely the two variables are related. A high positive correlation would indicate that exercise is strongly related to good health. In contrast, a low or negative correlation would suggest that exercise is not strongly related to good health. By understanding the relationship between variables, we can make informed decisions and take appropriate actions to achieve desired outcomes.")

    # Initialize the session state
    initialize_session_state()
    # Access the stored data in other files
    data = get_data()

    if data is not None:
        # Perform correlation analysis
        correlation(data)
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
