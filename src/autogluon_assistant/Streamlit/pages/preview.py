import streamlit as st
from st_aggrid import GridOptionsBuilder, AgGrid

def get_available_files():
    files = []
    if st.session_state.uploaded_files is not None:
        uploaded_files = st.session_state.uploaded_files
        files = list(uploaded_files.keys())
    return files


@st.fragment
def preview_dataset():
    """
        Displays a preview of the uploaded dataset in the Streamlit app.
   """
    st.markdown("""
        <h1 style='
            font-weight: light;
            padding-left: 20px;
            padding-right: 20px;
            margin-left:60px;
            font-size: 2em;
        '>
            Preview Dataset
        </h1>
    """, unsafe_allow_html=True)
    col1, col2, col3= st.columns([1, 22, 1])
    with col2:
        file_options = get_available_files()
        selected_file = st.selectbox("Preview Uploaded File", options=file_options,index = None,placeholder="Select the file to preview",label_visibility="collapsed")
        if st.session_state.uploaded_files is None:
            st.info("file not uploaded yet.", icon="ℹ️")
            return
        if selected_file is not None:
            st.markdown(f"""
            <div class="file-view-bar">
                <span class="file-view-label">Viewing File:</span> {selected_file}
            </div>
            """, unsafe_allow_html=True)
            gb = GridOptionsBuilder.from_dataframe(st.session_state.uploaded_files[selected_file]['df'])
            gb.configure_pagination()
            gridOptions = gb.build()
            AgGrid(st.session_state.uploaded_files[selected_file]['df'], gridOptions=gridOptions,
                   enable_enterprise_modules=False)


def main():
    preview_dataset()


if __name__ == "__main__":
    main()