import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import plotly.express as px
from session_state import get_data, initialize_session_state

# Set page title and icon
st.set_page_config(page_title="Clustering Analysis - Diagnostic Analysis Tool", page_icon=":bar_chart:", layout="wide")

# Write page header
def main():
    st.write("# Clustering Analysis")
    st.write("Clustering analysis is a type of unsupervised machine learning algorithm that is used to group similar data points together. The goal is to divide a dataset into groups or clusters based on similarities in the data, without being provided with any prior information about the grouping of the data.")
    st.write("In layman's terms, clustering analysis is like sorting laundry. Imagine you have a pile of clothes that you need to sort by color. You don't know how many different colors there are, but you can see that some of the clothes are similar in color to each other. Clustering analysis helps you to group similar clothes together without knowing beforehand how many different colors there are.")

    # Initialize session state
    initialize_session_state()

    # Get the uploaded dataset from the session state
    data = get_data()

    # Allow user to select features for clustering
    if data is not None:
        numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
        object_features = data.select_dtypes(include='object').columns.tolist()
        features = st.multiselect("Select features to cluster on", numeric_features + object_features)

        # Check if at least two features are selected
        if len(features) < 2:
            st.warning("Please select at least two features for clustering.")
        else:
            # Allow user to choose number of clusters
            n_clusters = st.slider("Select number of clusters", min_value=2, max_value=10)

            # Run clustering algorithm when user clicks 'Cluster Data' button
            if st.button("Cluster Data"):
                # Make a copy of the dataset
                data_copy = data.copy()

                # Preprocess data
                X = data_copy[features].copy()
                X_numeric = X.select_dtypes(include=[np.number])
                X_object = X.select_dtypes(include='object')
                imputer = SimpleImputer(strategy='mean')
                X_numeric = imputer.fit_transform(X_numeric)
                X_object = X_object.astype('category').apply(lambda x: x.cat.codes)

                # Combine preprocessed numeric and categorical data
                X_preprocessed = np.hstack((X_numeric, X_object))

                # Perform clustering
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                kmeans.fit(X_preprocessed)
                labels = kmeans.predict(X_preprocessed)
                centroids = kmeans.cluster_centers_

                # Count number of data points in each cluster
                cluster_counts = np.bincount(labels)

                # Visualize clusters
                fig = px.scatter_matrix(X, dimensions=features, color=labels, title="Clustering Results", color_continuous_scale='viridis')
                fig.update_layout(showlegend=True, legend_title_text='Cluster Labels', legend=dict(x=0.01, y=0.99))
                fig.update_xaxes(title_font=dict(color='black'), tickfont=dict(color='black'))  # Set x-axis label and tick color
                fig.update_yaxes(title_font=dict(color='black'), tickfont=dict(color='black'))  # Set y-axis label and tick color
                st.plotly_chart(fig, use_container_width=True)

                # Show summary of clustering results
                st.write("## Clustering Results")
                st.write("Number of Clusters: ", n_clusters)
                st.write("Total Number of Data Points: ", len(data_copy))
                
                # Calculate cluster statistics
                cluster_counts = np.bincount(labels)
                cluster_table = pd.DataFrame({'Cluster': range(n_clusters), 'Count': cluster_counts})
                st.write("Cluster Counts:")
                st.table(cluster_table)

                centroids_table = pd.DataFrame({'Cluster': range(n_clusters)})
                for i, centroid in enumerate(centroids):
                    for j, feature in enumerate(features):
                        centroids_table[feature] = centroid[j]
                st.write("Cluster Centroids:")
                st.table(centroids_table)

                # Add the "Cluster" column to the copied dataset
                data_copy.insert(0, "Cluster", labels)

                # Show which data point belongs to which cluster
                st.write("Data Points with Cluster Labels:")
                st.dataframe(data_copy)
            else:
                st.warning("Please click 'Cluster Data' to perform clustering analysis.")
    else:
        st.warning("No data found! Please upload a dataset.")

if __name__ == "__main__":
    main()
# Hide Streamlit footer note
hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
