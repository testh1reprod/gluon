import streamlit as st
import extra_streamlit_components as stx
st.set_page_config(page_title="AutoGluon Assistant",page_icon="https://pbs.twimg.com/profile_images/1373809646046040067/wTG6A_Ct_400x400.png", layout="wide",initial_sidebar_state="collapsed")
from pages.nav_bar import nav_bar
from pages.tutorial import main as tutorial
from pages.feature import main as feature
from pages.demo import main as demo
from pages.preview import main as preview
from pages.task import main as run



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
    if "train_file_name" not in st.session_state:
        st.session_state.train_file_name = None
    if "train_file_df" not in st.session_state:
        st.session_state.train_file_df = None
    if "test_file_name" not in st.session_state:
        st.session_state.test_file_name = None
    if "test_file_df" not in st.session_state:
        st.session_state.test_file_df = None
    if "sample_output_file_name" not in st.session_state:
        st.session_state.sample_output_file_name = None
    if "sample_output_file_df" not in st.session_state:
        st.session_state.sample_output_file_df = None

def get_manager():
    return stx.CookieManager()

cookie_manager = get_manager()


# st.subheader("All Cookies:")
# cookies = cookie_manager.get_all()
# st.write(cookies)
# c1, c2, c3 = st.columns(3)
#
# with c1:
#     st.subheader("Get Cookie:")
#     cookie = st.text_input("Cookie", key="0")
#     clicked = st.button("Get")
#     if clicked:
#         value = cookie_manager.get(cookie=cookie)
#         st.write(value)
# with c2:
#     st.subheader("Set Cookie:")
#     cookie = st.text_input("Cookie", key="1")
#     val = st.text_input("Value")
#     if st.button("Add"):
#         cookie_manager.set(cookie, val) # Expires in a day by default
# with c3:
#     st.subheader("Delete Cookie:")
#     cookie = st.text_input("Cookie", key="2")
#     if st.button("Delete"):
#         cookie_manager.delete(cookie)



def main():
    initial_session_state()
    nav_bar()
    tutorial()
    demo()
    feature()
    run()
    preview()
    # st.write(st.session_state)

    st.markdown("""
    <script src="https://code.jquery.com/jquery-3.3.1.slim.min.js" integrity="sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/popper.js@1.14.3/dist/umd/popper.min.js" integrity="sha384-ZMP7rVo3mIykV+2+9J3UJ46jBk0WLaUAdn689aCwoqbBJiSnjAK/l8WvCWPIPm49" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.1.3/dist/js/bootstrap.min.js" integrity="sha384-ChfqqxuZUCnJSK3+MXmPNIyE6ZbWh2IMqE241rYiqJxyMiZ6OW/JmZQ5stwEULTy" crossorigin="anonymous"></script>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
