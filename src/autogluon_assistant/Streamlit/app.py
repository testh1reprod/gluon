import streamlit as st
import extra_streamlit_components as stx
import streamlit.components.v1 as components
st.set_page_config(page_title="AutoGluon Assistant",page_icon="https://pbs.twimg.com/profile_images/1373809646046040067/wTG6A_Ct_400x400.png", layout="wide",initial_sidebar_state="collapsed")
from pages.nav_bar import nav_bar
from pages.tutorial import main as tutorial
from pages.feature import main as feature
from pages.demo import main as demo
from pages.preview import main as preview
from pages.task import main as run

# fontawesome
st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    """,
    unsafe_allow_html=True
)

# Bootstrap 4.1.3
st.markdown(
    """
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.1.3/dist/css/bootstrap.min.css" integrity="sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO" crossorigin="anonymous">
    """,unsafe_allow_html=True

)
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


reload_warning = """
<script>
  window.onbeforeunload = function () {

    return  "Are you sure want to LOGOUT the session ?";
}; 
</script>
"""

components.html(reload_warning,height=0)

st.markdown(
    f"""
               <style>
               .element-container:has(iframe[height="0"]) {{
                 display: None;
               }}
               </style>
           """, unsafe_allow_html=True
)

def initial_session_state():
    if 'config_overrides' not in st.session_state:
        st.session_state.config_overrides = []
    if "preset" not in st.session_state:
        st.session_state.preset = None
    if "time_limit" not in st.session_state:
        st.session_state.time_limit = None
    if "transformers" not in st.session_state:
        st.session_state.transformers = []
    if "llm" not in st.session_state:
        st.session_state.llm = None
    if "pid" not in st.session_state:
        st.session_state.pid = None
    if "logs" not in st.session_state:
        st.session_state.logs = ""
    if "process" not in st.session_state:
        st.session_state.process = None
    if 'clicked' not in st.session_state:
        st.session_state.clicked = False
    if "task_running" not in st.session_state:
        st.session_state.task_running = False
    if "output_file" not in st.session_state:
        st.session_state.output_file = None
    if "output_filename" not in st.session_state:
        st.session_state.output_filename = None
    if "task_description" not in st.session_state:
        st.session_state.task_description = ""
    if "return_code" not in st.session_state:
        st.session_state.return_code = None
    if "task_canceled" not in st.session_state:
        st.session_state.task_canceled = False
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = {}



def main():
    initial_session_state()
    nav_bar()
    tutorial()
    demo()
    feature()
    run()
    preview()

    st.markdown("""
    <script src="https://code.jquery.com/jquery-3.3.1.slim.min.js" integrity="sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/popper.js@1.14.3/dist/umd/popper.min.js" integrity="sha384-ZMP7rVo3mIykV+2+9J3UJ46jBk0WLaUAdn689aCwoqbBJiSnjAK/l8WvCWPIPm49" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.1.3/dist/js/bootstrap.min.js" integrity="sha384-ChfqqxuZUCnJSK3+MXmPNIyE6ZbWh2IMqE241rYiqJxyMiZ6OW/JmZQ5stwEULTy" crossorigin="anonymous"></script>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
