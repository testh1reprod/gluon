import streamlit as st
import pandas as pd
import base64
from pathlib import Path
from autogluon_assistant.Streamlit.style.style import styles,options
from streamlit_navigation_bar import st_navbar
import os
import uuid
import glob
import subprocess
import psutil
import streamlit.components.v1 as components
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.stylable_container import stylable_container


st.set_page_config(page_title="AutoGluon Assistant",page_icon="https://pbs.twimg.com/profile_images/1373809646046040067/wTG6A_Ct_400x400.png", layout="wide",initial_sidebar_state="collapsed")

# navigation header
page = st_navbar(["Run Autogluon","Dataset"], selected="Run Autogluon",styles=styles,options=options)

if page == "Dataset":
    st.switch_page("pages/preview.py")

CONFIG_DIR = '../../../config'
BASE_DATA_DIR = './user_data'

os.makedirs(BASE_DATA_DIR, exist_ok=True)

st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    """,
    unsafe_allow_html=True
)
with open('task_style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# image_path = "src/autogluon_assistant/Streamlit/images/logo.png"

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

def img_to_html(img_path):
    img_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
      img_to_bytes(img_path)
    )
    return img_html

def update_config_overrides():
    config_overrides = []
    if st.session_state.preset:
        preset_mapping = {"Best Quality": "best_quality", "High Quality": "high_quality",
                          "Good Quality": "good_quality", "Medium Quality": "medium_quality"}
        config_overrides.append(f"autogluon.predictor_fit_kwargs.presets={preset_mapping[st.session_state.preset]}")

    if st.session_state.time_limit:
        time_limit_mapping = {"30s": 30, "1 min": 60, "15 mins": 900, "30 mins": 1800, "1 hr": 3600, "2 hrs": 7200, "4 hrs": 14400}
        config_overrides.append(f"autogluon.predictor_fit_kwargs.time_limit={time_limit_mapping[st.session_state.time_limit]}")

    if st.session_state.transformers:
        transformers_mapping = {"OpenFE": "autogluon_assistant.transformer.OpenFETransformer",
                               "CAAFE": "autogluon_assistant.transformer.CAAFETransformer"}
        enabled_transformers = []
        for transformer in st.session_state.transformers:
            enabled_transformers.append(f"{{_target_:{transformers_mapping[transformer]}}}")
        config_overrides.append(f"feature_transformers=[{','.join(enabled_transformers)}]")
    else:
        config_overrides.append("feature_transformers=[]")

    if st.session_state.llm:
        llm_mapping = {"GPT 3.5-Turbo": "gpt-3.5-turbo", "GPT 4": "gpt-4-1106-preview",
                       "Claude 3 - Sonnet": "anthropic.claude-3-sonnet-20240229-v1:0"}
        config_overrides.append(f"llm.model={llm_mapping[st.session_state.llm]}")

    st.session_state.config_overrides = config_overrides

# These two functions are to save widget values in Session State to preserve them between pages
def store_value(key):
    st.session_state[key] = st.session_state["_"+key]
def load_value(key):
    st.session_state["_"+key] = st.session_state[key]

def set_params():
    col1, col2, col3, col4 = st.columns(4,gap='medium')
    with col1:
        config_autogluon_preset()
    with col2:
        config_time_limit()
    with col3:
        config_transformer()
    with col4:
        config_llm()
    update_config_overrides()
    add_vertical_space(2)

@st.fragment
def config_autogluon_preset():
    with st.container(border=True, height=150):
        col1, col2 = st.columns([1, 3])
        with col1:
            st.html(
                """
                 <div style="display: flex; align-items: center; justify-content: center; height: 100%;">
                <i class="fa-solid fa-gear" style="font-size: 50px; margin-top: 30px;color: #023e8a;"></i>
                </div>
                """
            )
        with col2:
            st.html(
                """
                <div style="display: flex; align-items: center; justify-content: center; height: 40px;padding-bottom: 16px;">
                    <h1 style="font-size: 20px;color:#023e8a; ">Autogluon Preset</h1>
                </div>
                """
            )
            preset_options = ["Best Quality", "High Quality", "Good Quality", "Medium Quality"]
            load_value("preset")
            st.selectbox("Autogluon Preset", index=None, placeholder="Autogluon Preset", options=preset_options, key="_preset",
                         on_change=store_value, args=["preset"],label_visibility="collapsed")

@st.fragment
def config_time_limit():
    with st.container(border=True, height=150):
        col1, col2 = st.columns([1, 3])
        with col1:
            st.html(
                """
                 <div style="display: flex; align-items: center; justify-content: center; height: 100%;">
                <i class="fa-solid fa-clock" style="font-size: 50px; margin-top: 30px;color: #023e8a;"></i>
                </div>
                """
            )
        with col2:
            st.html(
                """
                <div style="display: flex; align-items: center; justify-content: center; height: 40px;padding-bottom: 16px;">
                    <h1 style="font-size: 20px;color:#023e8a; ">Time Limit</h1>
                </div>
                """
            )
            time_limit_options = ["30s", "1 min", "15 mins", "30 mins", "1 hr", "2 hrs", "4 hrs"]
            load_value("time_limit")
            st.selectbox("Time Limit", index=None, placeholder="Time Limit", options=time_limit_options, key="_time_limit",on_change=store_value, args=["time_limit"],label_visibility="collapsed")

@st.fragment
def config_transformer():
    with st.container(border=True, height=150):
        col1, col2 = st.columns([1, 3])
        with col1:
            st.html(
                """
                 <div style="display: flex; align-items: center; justify-content: center; height: 100%;">
                <i class="fa-brands fa-buromobelexperte" style="font-size: 50px; margin-top: 30px;color: #023e8a;"></i>
                </div>
                """
            )
        with col2:
            st.html(
                """
                <div style="display: flex; align-items: center; justify-content: center; height: 40px;padding-bottom: 0;">
                    <h1 style="font-size: 19px;color:#023e8a; ">Feature Transformers</h1>
                </div>
                """
            )
            transformer_options = ["OpenFE", "CAAFE"]
            load_value("transformers")
            st.multiselect("Feature Transformers", placeholder="Feature Transformers", options=transformer_options,
                   key="_transformers", on_change=store_value, args=["transformers"],label_visibility="collapsed")

@st.fragment
def config_llm():
    with st.container(border=True, height=150):
        col1, col2 = st.columns([1, 3])
        with col1:
            st.html(
                """
                 <div style="display: flex; align-items: center; justify-content: center; height: 100%;">
                <i class="fa-solid fa-robot" style="font-size: 50px; margin-top: 30px;color: #023e8a;"></i>
                </div>
                """
            )
        with col2:
            st.html(
                """
                <div style="display: flex; align-items: center; justify-content: center; height: 40px;padding-bottom: 16px;">
                    <h1 style="font-size: 20px;color:#023e8a; ">LLM Model</h1>
                </div>
                """
            )
            llm_options = ["GPT 3.5-Turbo", "GPT 4", "Claude 3 - Sonnet"]
            load_value("llm")
            st.selectbox("Choose a LLM model", index=None, placeholder="Choose a LLM model", options=llm_options, key="_llm",
                 on_change=store_value, args=["llm"],label_visibility="collapsed")

def save_description_file(description):
    user_data_dir = get_user_data_dir()
    description_file = os.path.join(user_data_dir, "description.txt")
    with open(description_file, "w") as f:
        f.write(description)

def store_value_and_save_file(key):
    store_value(key)
    save_description_file(st.session_state.task_description)

def description_file_uploader():
    if "description_uploader_key" not in st.session_state:
        st.session_state.description_uploader_key = 0
    uploaded_file = st.file_uploader("Upload task description file", type="txt", key=st.session_state.description_uploader_key,help="Accepted file format: .txt",label_visibility="collapsed")
    if uploaded_file:
        task_description = uploaded_file.read().decode("utf-8")
        st.session_state.task_description = task_description
        save_description_file(st.session_state.task_description)
        st.session_state.description_uploader_key += 1
        st.rerun()

@st.fragment
def display_description():
    load_value("task_description")
    st.text_area(label='Task Description',placeholder="Enter your task description : ",value=st.session_state.task_description,key="_task_description",on_change=store_value_and_save_file, args=["task_description"],height=180,label_visibility="collapsed")



def display_header():
    image = './app/static/logo.png'
    st.markdown(
        f"""
        <div style="text-align: center;max-height: 150px;">
           <img src='{image}' alt='Logo' style='width:300px;'>
            <h1 style="margin: 0;color: #00428e; margin-bottom: 5px">Empower your ML with Zero Lines of Code</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    add_vertical_space(6)

def get_user_data_dir():
    # Generate a unique directory name for the user session if it doesn't exist
    if 'user_data_dir' not in st.session_state:
        unique_dir = str(uuid.uuid4())
        st.session_state.user_data_dir = os.path.join(BASE_DATA_DIR, unique_dir)
        os.makedirs(st.session_state.user_data_dir, exist_ok=True)

    return st.session_state.user_data_dir

def save_uploaded_file(file, file_path):
    if file.type == 'text/csv':
        with open(file_path, 'wb') as f:
            f.write(file.getbuffer())
    elif file.type == 'text/plain':
        with open(file_path, "w") as f:
            f.write(file.read().decode("utf-8"))

# This function is to make sure click on the download button does not reload the entire app (a temporary workaround)
@st.fragment
def show_download_button(data,file_name):
    st.download_button(label="Download the output file", data=data,file_name=file_name,mime="text/csv")

def show_cancel_task_button():
    try:
        if st.button("Stop Task", on_click=toggle_cancel_state):
            p = st.session_state.process
            print("Stopping the task ...")
            p.terminate()
            p.wait()
            st.session_state.process = None
            st.session_state.pid = None
            print("The Task has stopped")
            st.session_state.task_running = False
            st.rerun()
    except psutil.NoSuchProcess:
        st.session_state.task_running = False
        st.session_state.process = None
        st.session_state.pid = None
        st.error(f"No running task is found")
    except Exception as e:
        st.session_state.task_running = False
        st.session_state.process = None
        st.session_state.pid = None
        st.error(f"An error occurred: {e}")

def generate_output_file():
    if st.session_state.process is not None:
        process = st.session_state.process
        process.wait()
        st.session_state.task_running = False
        st.write("current running state",st.session_state.task_running)
        st.session_state.return_code = process.returncode
        st.session_state.process = None
        if st.session_state.return_code == 0:
            csv_files = glob.glob("*.csv")
            if csv_files:
                latest_csv = max(csv_files, key=os.path.getmtime)
                latest_csv_name = os.path.basename(latest_csv)
                df = pd.read_csv(latest_csv)
                st.session_state.output_file = df
                st.session_state.output_filename = latest_csv_name
                st.rerun()
            else:
                st.warning("No CSV file generated.")

# Run the autogluon-assistant command
def run_autogluon_assistant(config_dir, data_dir):
        command = ['autogluon-assistant', config_dir, data_dir]
        if st.session_state.config_overrides:
            command.extend(['--config-overrides', ' '.join(st.session_state.config_overrides)])
        st.session_state.output_file = None
        st.session_state.output_filename = None
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        st.session_state.process = process
        st.session_state.pid = process.pid

def show_logs():
    if st.session_state.logs:
        status_container = st.empty()
        log_container = st.empty()
        log_container.text_area("Real-Time Logs", st.session_state.logs, height=400)
        if st.session_state.return_code == 0:
            status_container.success("Task completed successfully!")
        else:
            status_container.error("Error detected in the process...Check the logs for more details")

def show_real_time_logs():
    if st.session_state.process is not None:
        process = st.session_state.process
        st.session_state.logs = ""
        progress = st.progress(0)
        task_stages = {
            "Task loaded!": 10,
            "Beginning AutoGluon training": 20,
            "Preprocessing data": 30,
            "User-specified model hyperparameters to be fit": 50,
            "Fitting model": 65,
            "AutoGluon training complete": 90,
        }
        logs_css = """
            <style>
                div[data-testid="stExpanderDetails"] {
                    height: 300px !important;
                    overflow-y: auto; 
                }
            </style>
            """
        st.markdown(logs_css, unsafe_allow_html=True)
        # # hide the iframe container
        st.markdown(
            f"""
                <style>
                .element-container:has(iframe[height="0"]) {{
                  display: None;
                }}
                </style>
            """, unsafe_allow_html=True
        )
        status_container = st.empty()
        status_container.info("Running Tasks...")
        log_container = st.empty()
        for line in process.stdout:
            print(line, end="")
            auto_scroll = """
            <script>
                const textAreas = parent.document.querySelectorAll('textarea[aria-label="Real-Time Logs"]');
                  if (textAreas.length > 0) 
                  {
                      const lastTextArea = textAreas[textAreas.length - 1];
                      lastTextArea.scrollTop = lastTextArea.scrollHeight;
                  }
            </script>
              """
            components.html(auto_scroll, height=0)
            st.session_state.logs += line
            if "exception" in line.lower():
                progress.empty()
                status_container.error("Error detected in the process...Check the logs for more details")
                st.session_state.output_file = None
                st.session_state.output_filename = None
                st.session_state.process = None
                st.session_state.pid = None
                st.session_state.task_running = False
                st.rerun()
            elif "Prediction complete" in line:
                progress.empty()
                status_container.success("Task completed successfully!")
                break
            else:
                for stage, progress_value in task_stages.items():
                    if stage.lower() in line.lower():
                        progress.progress(progress_value / 100)
                        status_container.info(stage)
                        break
            log_container.text_area("Real-Time Logs", st.session_state.logs, height=400)
    else:
        print("process is None")

def download_button():
    if st.session_state.output_file is not None:
        output_file = st.session_state.output_file
        output_filename = st.session_state.output_filename
        show_download_button(output_file.to_csv(index=False),output_filename)

def generate_task_file(user_data_dir):
    file_list = []
    if st.session_state.train_file_name is not None:
        file_list.append("train.csv")
    if st.session_state.test_file_name is not None:
        file_list.append("test.csv")
    if st.session_state.sample_output_file_name is not None:
        file_list.append("sample_output.csv")
    competition_files_path = os.path.join(user_data_dir, "task_files.txt")
    with open(competition_files_path, "w") as f:
        f.write("\n".join(file_list))

def train_uploader():
    train_file = st.file_uploader("Upload Train Dataset", key="train_file_uploader",label_visibility="collapsed")
    user_data_dir = get_user_data_dir()
    if train_file is not None:
        save_uploaded_file(train_file, os.path.join(user_data_dir, "train.csv"))
        train_df = pd.read_csv(train_file)
        st.session_state.train_file_name = train_file.name
        st.session_state.train_file_df = train_df
    if st.session_state.train_file_name is not None:
        with st.popover(st.session_state.train_file_name,use_container_width=True):
            st.write(st.session_state.train_file_df.head(10))
    add_vertical_space(1)

def test_uploader():
    test_file = st.file_uploader("Upload Test Dataset", key = 'test_file_uploader',label_visibility="collapsed")
    user_data_dir = get_user_data_dir()
    if test_file is not None:
        save_uploaded_file(test_file, os.path.join(user_data_dir, "test.csv"))
        test_df = pd.read_csv(test_file)
        st.session_state.test_file_name = test_file.name
        st.session_state.test_file_df = test_df
    if st.session_state.test_file_name is not None:
        with st.popover(st.session_state.test_file_name, use_container_width=True):
            st.write(st.session_state.test_file_df.head(10))
    add_vertical_space(1)

def sample_output_uploader():
    sample_output_file = st.file_uploader("Upload Sample Output Dataset (Optional)", key="sample_output_file_uploader",label_visibility="collapsed")
    user_data_dir = get_user_data_dir()
    if sample_output_file is not None:
        save_uploaded_file(sample_output_file, os.path.join(user_data_dir, "sample_output.csv"))
        sample_output_df = pd.read_csv(sample_output_file)
        st.session_state.sample_output_file_name = sample_output_file.name
        st.session_state.sample_output_file_df = sample_output_df
    if st.session_state.sample_output_file_name is not None:
        with st.popover(st.session_state.sample_output_file_name, use_container_width=True):
            st.write(st.session_state.sample_output_file_df.head(10))
    add_vertical_space(1)

def file_uploader():
    col1, col2, col3 = st.columns(3,gap="large")
    with col1:
        with stylable_container(key='train_file_uploader',css_styles="""
            {
                border: 1px solid rgba(49, 51, 63, 0.2);
                border-radius: 2rem;
                padding: calc(1em - 1px);
                background-color:#00428e;
                display: flex;
                flex-direction: column;
                align-items: center;
                box-shadow: 0 4px 8px gray;
            }
        """):
            train_uploader()
    with col2:
        with stylable_container(key='test_file_uploader', css_styles="""
                   {
                       border: 1px solid rgba(49, 51, 63, 0.2);
                       border-radius: 2rem;
                       padding: calc(1em - 1px);
                       background-color: #00428e;
                       display: flex;
                       flex-direction: column;
                       align-items: center;
                       box-shadow: 0 4px 8px gray;
                   }
               """):
            test_uploader()
    with col3:
        with stylable_container(key='sample_output_file_uploader', css_styles="""
                    {
                        border: 1px solid rgba(49, 51, 63, 0.2);
                        border-radius: 2rem;
                        padding: calc(1em - 1px);
                        background-color: #00428e;
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        box-shadow: 0 4px 8px gray;
                    }
                """):
            sample_output_uploader()




def toggle_running_state():
    st.session_state.task_running = True
    st.session_state.task_canceled = False

def toggle_cancel_state():
    st.session_state.task_canceled = True

def run_button():
    col1, col2, col3 = st.columns(3)
    with col2:
        user_data_dir = get_user_data_dir()
        if st.button(label="Run AutoGluon Assistant", on_click=toggle_running_state,
                     disabled=st.session_state.task_running):
            if st.session_state.train_file_name and st.session_state.test_file_name:
                    generate_task_file(user_data_dir)
                    run_autogluon_assistant(CONFIG_DIR, user_data_dir)
            else:
                st.warning("Please upload files before running the task.")
                st.session_state.task_running = False

def show_cancel_container():
    status_container = st.empty()
    status_container.info("Task has been cancelled")

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


def set_description():
    col1, col2 = st.columns([3,1],gap='large')
    with col1:
        st.html(
            """
            <div style="display: flex; align-items: center; justify-content: flex-start; height: 40px;padding-bottom: 16px;">
                <h1 style="font-size: 20px;color:#023e8a; ">Task Description</h1>
            </div>
            """
        )
        display_description()
    with col2:
        st.html(
            """
            <div style="display: flex; align-items: center; justify-content: flex-start; height: 40px;padding-bottom: 16px;">
                <h1 style="font-size: 20px;color:#023e8a; "></h1>
            </div>
            """
        )
        description_file_uploader()
    add_vertical_space(4)




def main():
    initial_session_state()
    display_header()
    set_params()
    set_description()
    file_uploader()
    run_button()
    if st.session_state.task_running:
        show_cancel_task_button()
    if st.session_state.task_running:
        show_real_time_logs()
    elif not st.session_state.task_running and not st.session_state.task_canceled:
        show_logs()
    elif st.session_state.task_canceled:
        show_cancel_container()


    generate_output_file()
    download_button()
    # st.write(st.session_state)



if __name__ == "__main__":
    main()
