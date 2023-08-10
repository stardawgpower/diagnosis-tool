import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
from session_state import initialize_session_state, get_data

def perform_path_analysis(dataset, latent_var, measured_var, month_title=None):
    # Calculate correlations between variables
    correlations = dataset.corr()

    # Create a directed graph
    G = nx.from_pandas_adjacency(correlations)

    # Create node colors and positions
    node_colors = ['blue' if node in measured_var else 'yellow' for node in G.nodes()]
    pos = nx.spring_layout(G, seed=42)

    # Create Plotly figure
    fig = go.Figure()

    # Create edge and node traces
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=1, color='gray'),
        hoverinfo='none',
        mode='lines')

    node_trace = go.Scatter(
        x=[],
        y=[],
        mode='markers',
        text=[str(latent_var)] + [str(var) for var in measured_var],
        hoverinfo='text',
        marker=dict(
            color=node_colors,
            size=15,
            line_width=2))

    # Add edges to the edge trace and edge annotations
    edge_colors = []

    for (u, v, attr) in G.edges(data=True):
        if (u in measured_var and v in latent_var) or (v in measured_var and u in latent_var):
            x_start, y_start = pos[u]
            x_end, y_end = pos[v]
            dx = x_end - x_start
            dy = y_end - y_start

            edge_colors.append(attr['weight'])

            fig.add_annotation(
                x=(x_start + x_end) / 2,
                y=(y_start + y_end) / 2,
                ax=(dx * 0.8 + x_start),
                ay=(dy * 0.8 + y_start),
                xref='x',
                yref='y',
                axref='x',
                ayref='y',
                showarrow=True,
                arrowhead=3,
                arrowsize=2,
                arrowwidth=1,
                arrowcolor='gray',
                text=f"Correlation:   {attr['weight']:.2f}",
                font=dict(size=10, color='black'))

            x_values = [x_start, x_end, None]
            y_values = [y_start, y_end, None]
            edge_trace['x'] += tuple(x_values)
            edge_trace['y'] += tuple(y_values)

    # Add node positions to the node trace
    node_trace['x'] = [pos[node][0] for node in G.nodes()]
    node_trace['y'] = [pos[node][1] for node in G.nodes()]

    # Add edge and node traces to the figure
    fig.add_trace(edge_trace)
    fig.add_trace(node_trace)

    # Set layout options
    title = 'Path Analysis'
    if month_title:
        title += f' - {month_title}'
    fig.update_layout(
        title=title,
        title_x=0.42,
        title_font=dict(size=20),
        showlegend=False,
        hovermode='closest',
        margin=dict(t=100, b=20),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False))

    # Show the plot
    st.plotly_chart(fig, use_container_width=True)

def main():
    st.set_page_config(page_title="Path Analysis (SEM) - Diagnostic Analysis Tool", page_icon=":bar_chart:", layout="wide")
    st.write("# Path Analysis")
    st.info("""
        Path analysis is a statistical technique used in structural equation modelling (SEM) to examine the relationships among variables.
        It allows researchers to explore direct and indirect effects and the total effects of variables on a target variable. 
        It helps to understand how different variables are connected.
        """)
    # Initialize session state and retrieve data
    initialize_session_state()
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
                selected_years1 = st.multiselect("Select years to compare (Widget 1)", years)

                # Filter the data based on the selected years and get unique months
                data_filtered1 = data[pd.DatetimeIndex(data[datetime_col]).year.isin(selected_years1)]

                # Remove columns with None values
                data_filtered1 = data_filtered1.dropna(axis=1, how="all")

                if datetime_col in data_filtered1.columns:
                    months1 = pd.DatetimeIndex(data_filtered1[datetime_col]).month_name().unique()
                    selected_month1 = st.selectbox("Select a month to analyze (Widget 1)", months1, key="widget1")

                    selected_years2 = st.multiselect("Select years to compare (Widget 2)", years)
                    # Filter the data based on the selected years and get unique months
                    data_filtered2 = data[pd.DatetimeIndex(data[datetime_col]).year.isin(selected_years2)]

                    # Remove columns with None values
                    data_filtered2 = data_filtered2.dropna(axis=1, how="all")
                    
                    if selected_years2:
                        months2 = pd.DatetimeIndex(data_filtered2[datetime_col]).month_name().unique()
                        selected_month2 = st.selectbox("Select a month to analyze (Widget 2)", months2, key="widget2")

                        num_cols = data_filtered1.select_dtypes(include=["float64", "int64"]).columns
                        latent_var = st.selectbox("Select Latent Variable", num_cols)

                        # Remove latent_var from the options for measured_var
                        measured_var_without_latent_var = [var for var in num_cols if var != latent_var]
                        measured_var = st.multiselect("Select Measured Variable(s)", measured_var_without_latent_var)

                    if st.button("Perform Path Analysis"):
                        filtered_df1 = data_filtered1[data_filtered1[datetime_col].dt.month_name() == selected_month1]
                        filtered_cols1 = [latent_var] + measured_var
                        filtered_data1 = filtered_df1[filtered_cols1].copy()
                        month_title1 = [selected_years1] + [selected_month1]
                        perform_path_analysis(filtered_data1, latent_var, measured_var, month_title1)

                        filtered_df2 = data_filtered2[data_filtered2[datetime_col].dt.month_name() == selected_month2]
                        filtered_cols2 = [latent_var] + measured_var
                        filtered_data2 = filtered_df2[filtered_cols2].copy()
                        month_title2 = [selected_years2] + [selected_month2]
                        perform_path_analysis(filtered_data2, latent_var, measured_var, month_title2)
                        st.info("In a path diagram, variables are represented as nodes, and the relationships between variables are depicted as arrows or paths. These paths indicate the direction and magnitude of the relationship between variables. The analysis aims to estimate the strength and significance of these relationships based on observed data.")
                        st.info("Here, the yellow node represents the latent variable( which can't be measured). Blue nodes represent the measured or observed variables.")

        else:
            num_cols = data.select_dtypes(include=["float64", "int64"]).columns
            latent_var = st.selectbox("Select Latent Variable", num_cols)

            # Remove latent_var from the options for measured_var
            measured_var_without_latent_var = [var for var in num_cols if var != latent_var]
            measured_var = st.multiselect("Select Measured Variable(s)", measured_var_without_latent_var)

            if st.button("Perform Path Analysis"):
                filtered_data = data[[latent_var] + measured_var].copy()
                perform_path_analysis(filtered_data, latent_var, measured_var)
                st.info("In a path diagram, variables are represented as nodes, and the relationships between variables are depicted as arrows or paths. These paths indicate the direction and magnitude of the relationship between variables. The analysis aims to estimate the strength and significance of these relationships based on observed data.")
                st.info("Here, the yellow node represents the latent variable( which can't be measured). Blue nodes represent the measured or observed variables.")
    else:
        st.warning("No data found! Please upload a dataset.")

if __name__ == "__main__":
    main()
