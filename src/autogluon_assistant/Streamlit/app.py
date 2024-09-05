import streamlit as st
from streamlit_navigation_bar import st_navbar

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

page = st_navbar(["Run Autogluon","Dataset"])

if page == "Run Autogluon":
    st.switch_page("pages/task.py")
if page == "Dataset":
    st.switch_page("pages/preview.py")

