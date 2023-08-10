import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from session_state import get_data, initialize_session_state

# Function to format the table with CSS styling
def format_table(table):
    styles = [{'selector': 'th', 'props': [('color', 'black')]}]
    return table.style.set_table_styles(styles)

def preprocess_data(data):
    label_encoders = {}  # Store the label encoder objects

    # Iterate over the columns
    for column in data.columns:
        if data[column].dtype in ['int64', 'float64']:
            # Handle missing values for numeric columns
            imputer = SimpleImputer(strategy='mean')
            data[column] = imputer.fit_transform(data[column].values.reshape(-1, 1))
            # Scale numeric features using StandardScaler
            scaler = StandardScaler()
            data[column] = scaler.fit_transform(data[column].values.reshape(-1, 1))
        elif data[column].dtype == 'object':
            # Handle missing values for categorical columns
            data[column] = data[column].fillna('Unknown')
            # Encode categorical variables using label encoding
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])
            label_encoders[column] = le  # Store the label encoder object

    # Additional preprocessing steps
    # Handling datetime data, imbalanced data, outliers, dimensionality reduction, etc.

    return data, label_encoders

def plot_confusion_matrix(cm, target_column, label_encoders):
    fig = px.imshow(cm, color_continuous_scale="Viridis", title="Confusion Matrix")
    fig.update_layout(width=800, height=800, showlegend=False, annotations=[
        go.layout.Annotation(
            x=i, y=j,
            text=str(cm[i, j]),
            font=dict(color="white", size=14),
            showarrow=False,
            xref="x", yref="y")
            for i in range(len(cm))
            for j in range(len(cm[0]))
    ])
    fig.update_layout(coloraxis_colorbar=dict(title="Count", titleside="right", thickness=15, len=0.75))

    x_axis_label = "Predicted Label"
    y_axis_label = "True Label"

    if label_encoders.get(target_column):
        # If the target column is categorical, use the original labels for the axis labels
        labels = label_encoders[target_column].inverse_transform(range(len(cm)))
        x_axis_label = f"Predicted {target_column}"
        y_axis_label = f"True {target_column}"

    fig.update_layout(xaxis_title=x_axis_label, yaxis_title=y_axis_label)
    fig.update_xaxes(title_font=dict(color='black'), tickfont=dict(color='black'))  # Set x-axis label and tick color
    fig.update_yaxes(title_font=dict(color='black'), tickfont=dict(color='black'))  # Set y-axis label and tick color
    
    return fig

def plot_feature_importances(importances, target_column, label_encoders):
    fig = px.bar(x=importances, y=importances.index, orientation='h', title="Feature Importances")

    x_axis_label = "Importance"
    y_axis_label = "Feature"

    if label_encoders.get(target_column):
        # If the target column is categorical, use the original column names for the y-axis labels
        y_axis_label = target_column

    fig.update_layout(xaxis_title=x_axis_label, yaxis_title=y_axis_label)
    fig.update_xaxes(title_font=dict(color='black'), tickfont=dict(color='black'))  # Set x-axis label and tick color
    fig.update_yaxes(title_font=dict(color='black'), tickfont=dict(color='black'))  # Set y-axis label and tick color
    
    return fig

def main():
    st.set_page_config(page_title="Classification Analysis - Diagnostic Analysis Tool", page_icon=":bar_chart:", layout="wide")
    st.write("# Classification Analysis")
    
    initialize_session_state()
    # Get the uploaded dataset from the session state
    data = get_data()
    
    if data is not None:
        # Make a copy of the data to prevent modifying the original dataset
        data_copy = data.copy()
        
        # Remove the datetime columns from the copied dataset
        data_copy.drop(columns=data_copy.select_dtypes(include=['datetime64[ns]']).columns, inplace=True)
        
        # Get the target column from the user
        target_column = st.selectbox("Select the target column", data_copy.columns)
        
        unique_values = data_copy[target_column].nunique()
        if unique_values <= 16:
            if data_copy[target_column].dtype == 'object':
                # Preprocess the data
                preprocessed_data, label_encoders = preprocess_data(data_copy)

                X = preprocessed_data.drop(columns=[target_column])
                y = preprocessed_data[target_column]

                # Split the data into training and test sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

                # Train a random forest classifier
                clf = RandomForestClassifier(n_estimators=100, random_state=0)
                clf.fit(X_train, y_train)

                # Make predictions on the test set
                y_pred = clf.predict(X_test)

                # Inverse transform the predicted labels
                y_test_original = label_encoders[target_column].inverse_transform(y_test)
                y_pred_original = label_encoders[target_column].inverse_transform(y_pred)

                # Display feature importances
                st.write("# Feature Importances")
                importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
                fig = plot_feature_importances(importances, target_column, label_encoders)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display Confusion matrix
                st.write("# Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig = plot_confusion_matrix(cm, target_column, label_encoders)
                st.plotly_chart(fig, use_container_width=True)

                # Generate classification report
                report = classification_report(y_test_original, y_pred_original, output_dict=True)
                df_report = pd.DataFrame(report).transpose()

                # Display Classification Report
                st.write("# Classification Report")
                st.table(format_table(df_report))

            else:
                # Handle numerical column with missing values
                missing_columns = data_copy.columns[data_copy.isnull().any()]
                if target_column in missing_columns:
                    # Handle missing values for the target column
                    imputer = SimpleImputer(strategy='mode')
                    data_copy[target_column] = imputer.fit_transform(data_copy[target_column].values.reshape(-1, 1))
                
                # Convert numerical column to object type in the copied dataset
                data_copy[target_column] = data_copy[target_column].astype('object')

                # Preprocess the data
                preprocessed_data, label_encoders = preprocess_data(data_copy)

                X = preprocessed_data.drop(columns=[target_column])
                y = preprocessed_data[target_column]

                # Split the data into training and test sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

                # Train a random forest classifier
                clf = RandomForestClassifier(n_estimators=100, random_state=0)
                clf.fit(X_train, y_train)

                # Make predictions on the test set
                y_pred = clf.predict(X_test)

                # Inverse transform the predicted labels
                y_test_original = label_encoders[target_column].inverse_transform(y_test)
                y_pred_original = label_encoders[target_column].inverse_transform(y_pred)

                # Display feature importances
                st.write("# Feature Importances")
                importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
                fig = plot_feature_importances(importances, target_column, label_encoders)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display Confusion matrix
                st.write("# Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig = plot_confusion_matrix(cm, target_column, label_encoders)
                st.plotly_chart(fig, use_container_width=True)

                # Generate classification report
                report = classification_report(y_test_original, y_pred_original, output_dict=True)
                df_report = pd.DataFrame(report).transpose()

                # Display Classification Report
                st.write("# Classification Report")
                st.table(format_table(df_report))
    
        else:
            st.warning("Please select the correct target column for classification analysis.")

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
