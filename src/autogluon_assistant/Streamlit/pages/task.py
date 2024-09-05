import streamlit as st
import pandas as pd
import base64
from pathlib import Path
from streamlit_navigation_bar import st_navbar
import os
import uuid
import subprocess
import glob


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
        # autogluon preset
        preset_options = ["Best Quality", "High Quality", "Good Quality", "Medium Quality"]
        load_value("preset")
        st.selectbox("Autogluon Preset",index=None, placeholder="Autogluon Preset", options=preset_options, key="_preset", on_change=store_value, args=["preset"])

        # Time limit
        time_limit_options = ["30s", "1 min", "15 mins", "30 mins" , "1 hr", "2 hrs", "4 hrs"]
        load_value("time_limit")
        st.selectbox("Time Limit",index=None, placeholder="Time Limit", options=time_limit_options, key="_time_limit",on_change=store_value, args=["time_limit"])

        # Feature Transformer
        transformer_options = ["OpenFE", "CAAFE"]
        load_value("transformers")
        st.multiselect("Feature Transformers", placeholder="Feature Transformers",options=transformer_options,key="_transformers",on_change=store_value, args=["transformers"])
        #  LLM
        llm_options = ["GPT 3.5-Turbo", "GPT 4", "Claude 3 - Sonnet"]
        load_value("llm")
        st.selectbox("Choose a LLM model",index=None, placeholder="Choose a LLM model",options=llm_options,key="_llm",on_change=store_value, args=["llm"])
    update_config_overrides()

def save_description_file(description):
    user_data_dir = get_user_data_dir()
    description_file = os.path.join(user_data_dir, "description.txt")
    with open(description_file, "w") as f:
        f.write(description)
    st.success(f"Description saved to {description_file}")

def store_value_and_save_file(key):
    store_value(key)
    save_description_file(st.session_state.task_description)

def display_description():
    if "task_description" not in st.session_state:
        st.session_state.task_description = ""
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

def save_uploaded_file(uploaded_file, directory):
    file_path = os.path.join(directory, uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# This function is to make sure click on the download button does not reload the entire app (a temporary workaround)
@st.fragment
def show_download_button(data,file_name):
    st.download_button(label="Download the output file", data=data,file_name=file_name,mime="text/csv")


# Run the autogluon-assistant command
def run_autogluon_assistant(config_dir, data_dir):
    command = ['autogluon-assistant', config_dir, data_dir]
    if st.session_state.config_overrides:
        command.extend(['--config-overrides', ' '.join(st.session_state.config_overrides)])
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # Stream the output line by line in real-time
    log_placeholder = st.empty()
    logs = ""
    for line in process.stdout:
        logs += line
        print(line, end="")
        log_placeholder.text_area("Real-Time Logs", logs, height=400)
    process.wait()  # Wait for the process to complete
    if process.returncode == 0:
        st.success("Task completed successfully!")
        csv_files = glob.glob("*.csv")
        if csv_files:
            latest_csv = max(csv_files, key=os.path.getmtime)
            df = pd.read_csv(latest_csv)
            st.write("Generated CSV file:")
            st.dataframe(df)
            show_download_button(df.to_csv(index=False),os.path.basename(latest_csv))
        else:
            st.warning("No CSV file generated.")
    else:
        st.error("Task failed. Check the logs for more details.")



def file_uploader():
    uploaded_files = st.file_uploader("Select the training dataset", accept_multiple_files=True)
    user_data_dir = get_user_data_dir()
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = {}
    if uploaded_files:
        st.write("Uploaded Files:")
        for file in uploaded_files:
            st.write(f"- {file.name}")
            if file.type == 'text/csv':
                # Check if the file is already in the session state
                if file.name in st.session_state.uploaded_files:
                    df = st.session_state.uploaded_files[file.name]
                else:
                    df = pd.read_csv(file)
                    save_uploaded_file(file, user_data_dir)
                    st.session_state.uploaded_files[file.name] = df
                with st.expander("Show/Hide File Data"):
                    st.write(df.head())
                # else:
                    # st.warning(f"File '{file.name}' is already uploaded")
    # Button to run AutoGluon Assistant
    if st.button("Run AutoGluon Assistant"):
        if uploaded_files:
            run_autogluon_assistant(CONFIG_DIR, user_data_dir)
        else:
            st.warning("Please upload files before running the task.")






set_params()
display_header()
st.write("")
st.write("")

display_description()
file_uploader()
