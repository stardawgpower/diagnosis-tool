import streamlit as st
from PIL import Image
from dt_auto import read_csv
from pandas.api.types import is_numeric_dtype
from session_state import initialize_session_state, set_data

# Create a home page
def home():
    st.set_page_config(page_title="Diagnostic Analysis Tool", page_icon=":bar_chart:", layout="wide")
    st.title("Welcome to Diagnostic Analysis Tool!")
    st.write("This tool lets you quickly analyse your data and gain insights to inform your decision-making. Upload your CSV file using the 'Browse files' button below. Once your data is loaded, you can use the various features and tools to visualise and explore your data, including generating descriptive statistics, creating charts and graphs, finding influencing factors and more.") 
    st.write("Our user-friendly interface makes it easy to work with your data, even if you're new to data analysis. Start using our Diagnostic Analysis application today and unlock the full potential of your data!")    
    image = Image.open("images/types.jpeg")
    new_image = image.resize((900,500))
    st.image(new_image)
    uploaded_file = st.file_uploader("Upload your dataset here ✅", type=["csv"])
    
    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Read the uploaded file into a pandas DataFrame
        data = read_csv(uploaded_file)

        # Drop empty columns
        empty_columns = data.columns[data.isnull().all()]
        data.drop(columns=empty_columns, inplace=True)

        # Drop columns with the same value in each row
        same_valued_columns = []
        for col in data.columns:
            if is_numeric_dtype(data[col]):
                if data[col].nunique() == 1:
                    same_valued_columns.append(col)
            else:
                if data[col].nunique() == 1 and data[col].notna().all():
                    same_valued_columns.append(col)
        data.drop(columns=same_valued_columns, inplace=True)

        # Add code to drop S.No.
        
        # Renaming the datetime columns
        rename_mapping = {}
        for col in data.columns:
            if data[col].dtype == "datetime64[ns]":
                new_name = f"{col}_dt"
                rename_mapping[col] = new_name

        data = data.rename(columns=rename_mapping)

        # Store the data in a Streamlit session state variable
        set_data(data)
        
        # Display a success message
        st.success("File uploaded successfully!")

# Initialize the session state
initialize_session_state()

# Run the home page
home()
    
# Hide Streamlit footer note
hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
