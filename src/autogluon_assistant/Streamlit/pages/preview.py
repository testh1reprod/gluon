import streamlit as st
from streamlit_navigation_bar import st_navbar
from st_aggrid import GridOptionsBuilder, AgGrid
from autogluon_assistant.Streamlit.style.style import styles,options
import hydralit_components as hc

st.set_page_config(page_title="Preview Dataset",initial_sidebar_state="collapsed",layout="wide",page_icon="https://pbs.twimg.com/profile_images/1373809646046040067/wTG6A_Ct_400x400.png")
# Navigation header
page = st_navbar(["Run Autogluon","Dataset"], selected="Dataset",styles=styles,options=options)

if page == "Run Autogluon":
    st.switch_page("pages/task.py")

with open('preview_style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def preview_dataset():
    col1, col2 = st.columns([1, 3], gap='small')
    with col1:
        option_data = [
            {'icon': "bi bi-file-earmark-text", 'label': "Train File"},
            {'icon': "bi bi-file-earmark-spreadsheet", 'label': "Test File"},
            {'icon': "bi bi-file-earmark-bar-graph", 'label': "Sample Output File"},
            {'icon': "bi bi-file-earmark-arrow-down", 'label': "Output File"}
        ]
        over_theme = {}
        font_fmt = {'font-class': 'h2', 'font-size': '150%'}
        selected_file = hc.option_bar(option_definition=option_data, key='PrimaryOption',
                                      override_theme=over_theme, font_styling=font_fmt, horizontal_orientation=False)
    with col2:
        selected_file_container = st.container()
        with selected_file_container:
            if selected_file == "Train File" and st.session_state.train_file_df is not None:
                st.subheader(f"Preview of '{st.session_state.train_file_name}'")
                gb = GridOptionsBuilder.from_dataframe(st.session_state.train_file_df)
                gb.configure_pagination()
                gridOptions = gb.build()
                AgGrid(st.session_state.train_file_df, gridOptions=gridOptions,
                       enable_enterprise_modules=False)
            elif selected_file == "Test File" and st.session_state.test_file_df is not None:
                st.subheader(f"Preview of '{st.session_state.test_file_name}'")
                gb = GridOptionsBuilder.from_dataframe(st.session_state.test_file_df)
                gb.configure_pagination()
                gridOptions = gb.build()
                AgGrid(st.session_state.test_file_df, gridOptions=gridOptions,
                       enable_enterprise_modules=False)
            elif selected_file == "Sample Output File" and st.session_state.sample_output_file_df is not None:
                st.subheader(f"Preview of '{st.session_state.sample_output_file_name}'")
                gb = GridOptionsBuilder.from_dataframe(st.session_state.sample_output_file_df)
                gb.configure_pagination()
                gridOptions = gb.build()
                AgGrid(st.session_state.sample_output_file_df, gridOptions=gridOptions,
                       enable_enterprise_modules=False)
            elif selected_file == "Output File" and st.session_state.output_file is not None:
                st.subheader(f"Preview of '{st.session_state.output_filename}'")
                gb = GridOptionsBuilder.from_dataframe(st.session_state.output_file)
                gb.configure_pagination()
                gridOptions = gb.build()
                AgGrid(st.session_state.output_file, gridOptions=gridOptions,
                       enable_enterprise_modules=False)
            else:
                st.info("file not uploaded yet.", icon="ℹ️")


def main():
    preview_dataset()


if __name__ == "__main__":
    main()