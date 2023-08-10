import streamlit as st

def initialize_session_state():
    if 'data' not in st.session_state:
        st.session_state['data'] = None

def set_data(data):
    st.session_state['data'] = data

def get_data():
    return st.session_state['data']


