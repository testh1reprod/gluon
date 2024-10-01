import streamlit as st
from st_aggrid import GridOptionsBuilder, AgGrid

def get_available_files():
    files = []
    file_types = {
        'train_file_name': 'Train File',
        'test_file_name': 'Test File',
        'sample_output_file_name': 'Sample Output File',
        'output_filename': 'Output File'
    }
    for key, label in file_types.items():
        if st.session_state[key] is not None:
            files.append(f"{label}: {st.session_state[key]}")
    return files


@st.fragment
def preview_dataset():
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
        if st.session_state.train_file_name is None and st.session_state.test_file_name is None and st.session_state.sample_output_file_name is None:
            st.info("file not uploaded yet.", icon="ℹ️")
            return
        if selected_file is not None:
            file_type, file_name = selected_file.split(': ', 1)
            st.markdown(f"""
            <div class="file-view-bar">
                <span class="file-view-label">Viewing File:</span> {file_name}
            </div>
            """, unsafe_allow_html=True)
            if file_type == "Train File" and st.session_state.train_file_df is not None:
                gb = GridOptionsBuilder.from_dataframe(st.session_state.train_file_df)
                gb.configure_pagination()
                gridOptions = gb.build()
                AgGrid(st.session_state.train_file_df, gridOptions=gridOptions,
                       enable_enterprise_modules=False)
            elif file_type == "Test File" and st.session_state.test_file_df is not None:
                gb = GridOptionsBuilder.from_dataframe(st.session_state.test_file_df)
                gb.configure_pagination()
                gridOptions = gb.build()
                AgGrid(st.session_state.test_file_df, gridOptions=gridOptions,
                       enable_enterprise_modules=False)
            elif file_type == "Sample Output File" and st.session_state.sample_output_file_df is not None:
                gb = GridOptionsBuilder.from_dataframe(st.session_state.sample_output_file_df)
                gb.configure_pagination()
                gridOptions = gb.build()
                AgGrid(st.session_state.sample_output_file_df, gridOptions=gridOptions,
                       enable_enterprise_modules=False)
            elif file_type == "Output File" and st.session_state.output_filename is not None:
                st.subheader(f"Preview of '{st.session_state.output_filename}'")
                gb = GridOptionsBuilder.from_dataframe(st.session_state.output_file)
                gb.configure_pagination()
                gridOptions = gb.build()
                AgGrid(st.session_state.output_file, gridOptions=gridOptions,
                       enable_enterprise_modules=False)

def main():
    preview_dataset()


if __name__ == "__main__":
    main()