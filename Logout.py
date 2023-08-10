import streamlit as st
from PIL import Image
from streamlit_extras.switch_page_button import switch_page

st.set_page_config(page_title = "Login")
st.markdown("<style> ul {display: none;} </style>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        /* Add your CSS changes here */
        /*color: red;*/
        /*background-color : pink;*/
        /*font-weight: bold;*/
        display:none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

image = Image.open("images/diagnostic.jpeg")
st.image(image,width=700)

button_style = """
        <style>
        .row-widget{
            width: 700px;
        }
        </style>
        """
st.markdown(button_style, unsafe_allow_html=True)
button_style = """
        <style>
        .block-container{
            
            width: 700px;
                display: flex;
    justify-content: center;
        }
        </style>
        """
st.markdown(button_style, unsafe_allow_html=True)

username = st.text_input("User ID")
password = st.text_input("Password", type="password")
login_button = st.button("Login")
page_bg_color = '''
<style>
body {
background-color:;
}
</style>'''
st.markdown(page_bg_color, unsafe_allow_html=True)
button_style = """
<style>
.stButton > button {
    color: white;
    font-weight: bold;
    top: 10px;
    left: 600px;
    position: relative;
    background: blue;
    width: 100px;
    height: 40px;
}
</style>"""
st.markdown(button_style, unsafe_allow_html=True)

if login_button:
       
        if username == "example" and password == "password":
            switch_page("Home")
        else:
            st.error("Invalid username or password")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
