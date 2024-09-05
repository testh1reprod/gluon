import streamlit as st
from streamlit_navigation_bar import st_navbar
from st_aggrid import GridOptionsBuilder, AgGrid


st.set_page_config(initial_sidebar_state="collapsed")

# Navigation header
page = st_navbar(["Run Autogluon","Dataset"], selected="Dataset")

if page == "Run Autogluon":
    st.switch_page("pages/task.py")

def preview_dataset():
    if 'uploaded_files' in st.session_state and  st.session_state.uploaded_files:
        uploaded_files = st.session_state.uploaded_files
        st.write("Preview of Uploaded Files:")
        file_names = list(uploaded_files.keys())
        selected_file = st.selectbox("Select a file to preview:", file_names)
        gb = GridOptionsBuilder.from_dataframe(st.session_state.uploaded_files[selected_file])
        gb.configure_pagination()
        gridOptions = gb.build()
        st.write(f"### Preview of '{selected_file}'")
        AgGrid(st.session_state.uploaded_files[selected_file],gridOptions=gridOptions,enable_enterprise_modules=False)


    else:
        st.write("No files uploaded. Please upload files on the Run Task page.")

preview_dataset()
