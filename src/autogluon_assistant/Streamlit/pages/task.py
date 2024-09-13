import streamlit as st
import pandas as pd
import base64
from pathlib import Path
from streamlit_navigation_bar import st_navbar
import os
import uuid
import glob
from io import StringIO
import subprocess
import psutil
import streamlit.components.v1 as components

st.set_page_config(page_title="AutoGluon Assistant",page_icon="https://pbs.twimg.com/profile_images/1373809646046040067/wTG6A_Ct_400x400.png", layout="wide")

# navigation header
page = st_navbar(["Run Autogluon","Dataset"], selected="Run Autogluon")

if page == "Dataset":
    st.switch_page("pages/preview.py")

CONFIG_DIR = '../../../config'
BASE_DATA_DIR = './user_data'

os.makedirs(BASE_DATA_DIR, exist_ok=True)



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

    with st.sidebar:
        st.image("./images/logo.png")
        config_autogluon_preset()
        config_time_limit()
        config_transformer()
        config_llm()
    update_config_overrides()

@st.fragment
def config_autogluon_preset():
    preset_options = ["Best Quality", "High Quality", "Good Quality", "Medium Quality"]
    load_value("preset")
    st.selectbox("Autogluon Preset", index=None, placeholder="Autogluon Preset", options=preset_options, key="_preset",
                 on_change=store_value, args=["preset"])

@st.fragment
def config_time_limit():
    time_limit_options = ["30s", "1 min", "15 mins", "30 mins", "1 hr", "2 hrs", "4 hrs"]
    load_value("time_limit")
    st.selectbox("Time Limit", index=None, placeholder="Time Limit", options=time_limit_options, key="_time_limit",
                 on_change=store_value, args=["time_limit"])

@st.fragment
def config_transformer():
    transformer_options = ["OpenFE", "CAAFE"]
    load_value("transformers")
    st.multiselect("Feature Transformers", placeholder="Feature Transformers", options=transformer_options,
                   key="_transformers", on_change=store_value, args=["transformers"])

@st.fragment
def config_llm():
    llm_options = ["GPT 3.5-Turbo", "GPT 4", "Claude 3 - Sonnet"]
    load_value("llm")
    st.selectbox("Choose a LLM model", index=None, placeholder="Choose a LLM model", options=llm_options, key="_llm",
                 on_change=store_value, args=["llm"])

def save_description_file(description):
    user_data_dir = get_user_data_dir()
    description_file = os.path.join(user_data_dir, "description.txt")
    with open(description_file, "w") as f:
        f.write(description)
    st.success(f"Description saved to {description_file}")

def store_value_and_save_file(key):
    store_value(key)
    save_description_file(st.session_state.task_description)

@st.fragment
def display_description():
    load_value("task_description")
    st.text_area(label='Task Description',placeholder="Enter your task description : ",key="_task_description",on_change=store_value_and_save_file, args=["task_description"])


def display_header():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("./images/logo.png")
    st.markdown(
        """
        <div style="text-align: center;max-height: 150px;">
            <h1 style="margin: 0;color: #1779be">Empower your ML with Zero Lines of Code</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

def get_user_data_dir():
    # Generate a unique directory name for the user session if it doesn't exist
    if 'user_data_dir' not in st.session_state:
        unique_dir = str(uuid.uuid4())
        st.session_state.user_data_dir = os.path.join(BASE_DATA_DIR, unique_dir)
        os.makedirs(st.session_state.user_data_dir, exist_ok=True)

    return st.session_state.user_data_dir

def save_uploaded_file(file, directory):
    file_path = os.path.join(directory, file.name)
    if file.type == 'text/csv':
        with open(file_path, 'wb') as f:
            f.write(file.getbuffer())
    elif file.type == 'text/plain':
        with open(file_path, "w") as f:
            f.write(file.read().decode("utf-8"))
    return file_path

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
                /* Set height for the stExpanderDetails */
                div[data-testid="stExpanderDetails"] {
                    height: 300px !important;
                    background-color: lightgrey;
                    overflow-y: auto; /* Ensures content scrolls if it overflows */
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
                break
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
    competition_files_path = os.path.join(user_data_dir, "task_files.txt")
    csv_file_names = [file_name for file_name in st.session_state.uploaded_files.keys() if file_name.endswith(".csv")]
    with open(competition_files_path, "w") as f:
        f.write("\n".join(csv_file_names))

def file_uploader():
    uploaded_files = st.file_uploader("Select the training dataset", accept_multiple_files=True)
    user_data_dir = get_user_data_dir()
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = {}
    if uploaded_files:
        st.markdown('''
            <style>
                .stFileUploaderFile {display: none}
            <style>''',
                    unsafe_allow_html=True)
        for file in uploaded_files:
            if file.name not in st.session_state.uploaded_files.keys():
                if file.type == 'text/csv':
                    df = pd.read_csv(file)
                    save_uploaded_file(file, user_data_dir)
                    st.session_state.uploaded_files[file.name] = df
                elif file.type == 'text/plain':
                    save_uploaded_file(file, user_data_dir)
                    stringio = StringIO(file.getvalue().decode("utf-8"))
                    st.session_state.uploaded_files[file.name] = stringio
    if st.session_state.uploaded_files:
        st.write("Uploaded Files:")
        for file_name, file_contents in st.session_state.uploaded_files.items():
            st.write(f"- {file_name}")
            if file_name.endswith('.csv'):
                df = file_contents
                with st.expander("Show/Hide File Data"):
                    st.write(df.head())
            elif file_name.endswith('.txt'):
                with st.expander("Show/Hide File Data"):
                    st.text_area(label=file_name,value=file_contents.getvalue(),label_visibility="collapsed")

def toggle_running_state():
    st.session_state.task_running = True
    st.session_state.task_canceled = False

def toggle_cancel_state():
    st.session_state.task_canceled = True

def run_button():
    user_data_dir = get_user_data_dir()
    if st.button(label="Run AutoGluon Assistant", on_click=toggle_running_state,
                 disabled=st.session_state.task_running):
        if 'uploaded_files' in st.session_state and st.session_state.uploaded_files:
            generate_task_file(user_data_dir)
            run_autogluon_assistant(CONFIG_DIR, user_data_dir)
        else:
            st.warning("Please upload files before running the task.")
            st.session_state.task_running = False

def show_cancel_container():
    status_container = st.empty()
    status_container.info("Task has been cancelled")

def initial_session_state():
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

def main():
    initial_session_state()
    set_params()
    display_header()
    st.write("")
    st.write("")
    display_description()
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
    st.write(st.session_state)


if __name__ == "__main__":
    main()
