import streamlit as st
from streamlit_navigation_bar import st_navbar
from style.style import styles,options

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)
pages = ["Home","Run Autogluon","Dataset"]

page = st_navbar(pages,styles=styles,options=options)

if page == "Run Autogluon":
    st.switch_page("pages/task.py")
if page == "Dataset":
    st.switch_page("pages/preview.py")
if page == "Home":
    st.switch_page("pages/home.py")

