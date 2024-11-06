import streamlit as st
import os
import pandas as pd
import uuid
import subprocess
import psutil
import streamlit.components.v1 as components
from streamlit_extras.add_vertical_space import add_vertical_space



CONFIG_DIR = '../../../config'
BASE_DATA_DIR = './user_data'

os.makedirs(BASE_DATA_DIR, exist_ok=True)
def update_config_overrides():
    """
        Update the configuration overrides based on the current session state.
    """
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

def store_value(key):
    """
        Store a value in the session state.

        Args:
            key (str): The key to store the value under in the session state.
    """
    st.session_state[key] = st.session_state["_"+key]

def load_value(key):
    """
        Load a value from the session state into a temporary key.

        Args:
            key (str): The key to load the value from in the session state.
    """
    st.session_state["_"+key] = st.session_state[key]
@st.fragment
def config_autogluon_preset():
    preset_options = ["Best Quality", "High Quality", "Good Quality", "Medium Quality"]
    load_value("preset")
    st.selectbox("Autogluon Preset", index=None, placeholder="Autogluon Preset", options=preset_options, key="_preset",
                 on_change=store_value, args=["preset"],label_visibility="collapsed")

@st.fragment
def config_time_limit():

    time_limit_options = ["30s", "1 min", "15 mins", "30 mins", "1 hr", "2 hrs", "4 hrs"]
    load_value("time_limit")
    st.selectbox("Time Limit", index=None, placeholder="Time Limit", options=time_limit_options, key="_time_limit",on_change=store_value, args=["time_limit"],label_visibility="collapsed")

@st.fragment
def config_transformer():
    transformer_options = ["OpenFE", "CAAFE"]
    load_value("transformers")
    st.multiselect("Feature Transformers", placeholder="Feature Transformers", options=transformer_options,
           key="_transformers", on_change=store_value, args=["transformers"],label_visibility="collapsed")

@st.fragment
def config_llm():
    llm_options = ["GPT 3.5-Turbo", "GPT 4", "Claude 3 - Sonnet"]
    load_value("llm")
    st.selectbox("Choose a LLM model", index=None, placeholder="Choose a LLM model", options=llm_options, key="_llm",
         on_change=store_value, args=["llm"],label_visibility="collapsed")

def save_description_file(description):
    """
        Save the task description to a file in the user's data directory.

        Args:
            description (str): The task description to save.
    """
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
    st.text_area(label='Dataset Description',placeholder="Enter your task description : ",key="_task_description",on_change=store_value_and_save_file, args=["task_description"],height=250)

def get_user_session_id():
    """
        Get or generate a unique user session ID.

        Returns:
            str: A unique identifier for the current user session.
    """
    if 'user_session_id' not in st.session_state:
        st.session_state.user_session_id = str(uuid.uuid4())
    return st.session_state.user_session_id


def generate_output_filename():
    """
        Generate a unique output filename based on the user session ID and current timestamp.

        Returns:
            str: A unique filename for the output CSV file.
    """
    user_session_id = get_user_session_id()
    unique_id = user_session_id[:8]
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f"output_{unique_id}_{timestamp}.csv"
    return output_filename

def get_user_data_dir():
    """
        Get or create a unique directory for the current user session.

        Returns:
            str: The path to the user's data directory.
    """
    if 'user_data_dir' not in st.session_state:
        unique_dir = st.session_state.user_session_id
        st.session_state.user_data_dir = os.path.join(BASE_DATA_DIR, unique_dir)
        os.makedirs(st.session_state.user_data_dir, exist_ok=True)
    return st.session_state.user_data_dir

def clear_directory(directory):
    """
    Before saving the files to the user's directory, clear the directory first
    """
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            os.remove(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def save_uploaded_file(file, directory):
    """
        Save an uploaded file to the specified directory.

        Args:
            file (UploadedFile): The file uploaded by the user.
            directory (str): The directory to save the file in.
    """
    file_path = os.path.join(directory, file.name)
    if not os.path.exists(file_path):
        with open(file_path, 'wb') as f:
            f.write(file.getbuffer())

@st.fragment
def show_download_button(data,file_name):
    st.download_button(label="💾&nbsp;&nbsp;Download the output file", data=data,file_name=file_name,mime="text/csv")

def show_cancel_task_button():
    try:
        if st.button("⏹️&nbsp;&nbsp;Stop Task", on_click=toggle_cancel_state):
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
    """
        Generate and store the output file after task completion.

    """
    if st.session_state.process is not None:
        process = st.session_state.process
        process.wait()
        st.session_state.task_running = False
        st.session_state.return_code = process.returncode
        st.session_state.process = None
        if st.session_state.return_code == 0:
            output_filename = st.session_state.output_filename
            if output_filename:
                df = pd.read_csv(output_filename)
                st.session_state.output_file = df
                st.rerun()
            else:
                st.error(f"CSV file not found: {output_filename}")


def run_autogluon_assistant(config_dir, data_dir):
    """
       Run the AutoGluon assistant with the specified configuration and data directories.

       This function constructs the command to run the AutoGluon assistant, including
       any config overrides, and starts the process. The process object and output
       filename are stored in the session state.

       Args:
           config_dir (str): The path to the configuration directory.
           data_dir (str): The path to the data directory.
    """
    command = ['autogluon-assistant', config_dir, data_dir]
    if st.session_state.config_overrides:
        command.extend(['--config-overrides', ' '.join(st.session_state.config_overrides)])
    output_filename = generate_output_filename()
    command.extend(['--output-filename', output_filename])
    st.session_state.output_file = None
    st.session_state.output_filename = output_filename
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    st.session_state.process = process
    st.session_state.pid = process.pid

def show_logs():
    """
       Display logs and task status when task is finished.
    """
    if st.session_state.logs:
        status_container = st.empty()
        log_container = st.empty()
        log_container.text_area("Real-Time Logs", st.session_state.logs, height=400)
        if st.session_state.return_code == 0:
            status_container.success("Task completed successfully!")
        else:
            status_container.error("Error detected in the process...Check the logs for more details")

def show_real_time_logs():
    """
        Display real-time logs and progress updates for the running task.
    """
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
        # hide the iframe container
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
    """
        Create and display a download button for the output file.
    """
    if st.session_state.output_file is not None and st.session_state.task_running is False:
        output_file = st.session_state.output_file
        output_filename = st.session_state.output_filename
        show_download_button(output_file.to_csv(index=False),output_filename)

def generate_task_file(user_data_dir):
    """
        Generate a task file containing names of uploaded CSV files.

        Args:
            user_data_dir (str): Path to the user's data directory.
    """
    competition_files_path = os.path.join(user_data_dir, "task_files.txt")
    csv_file_names = [file_name for file_name in st.session_state.uploaded_files.keys() if file_name.endswith(".csv")]
    with open(competition_files_path, "w") as f:
        f.write("\n".join(csv_file_names))

def save_all_files(user_data_dir):
    """
    When the task starts to run, save all the user's uploaded files to user's directory
    """
    clear_directory(user_data_dir)
    for file_name, file_data in st.session_state.uploaded_files.items():
        save_uploaded_file(file_data['file'], user_data_dir)

def file_uploader():
    """
        Handle file uploads
    """
    st.markdown("#### Upload Dataset")
    uploaded_files = st.file_uploader("Select the dataset", accept_multiple_files=True,label_visibility="collapsed",type=['csv','xlsx'])
    st.session_state.uploaded_files = {}
    for file in uploaded_files:
        df = pd.read_csv(file)
        st.session_state.uploaded_files[file.name] = {
            'file': file,
            'df': df
        }


def toggle_running_state():
    st.session_state.task_running = True
    st.session_state.task_canceled = False

def toggle_cancel_state():
    st.session_state.task_canceled = True

def run_button():
    """
        Create and handle the "Run" button for starting the AutoGluon task.
    """
    user_data_dir = get_user_data_dir()
    if st.button(label="🔘&nbsp;&nbsp;Run!", on_click=toggle_running_state,
                 disabled=st.session_state.task_running):
        if st.session_state.uploaded_files is not None:
                save_all_files(user_data_dir)
                generate_task_file(user_data_dir)
                run_autogluon_assistant(CONFIG_DIR, user_data_dir)
        else:
            st.warning("Please upload files before running the task.")
            st.session_state.task_running = False

def show_cancel_container():
    """
        Display a cancellation message when the task is cancelled.
    """
    status_container = st.empty()
    status_container.info("Task has been cancelled")

def set_description():
    """
        Set up the description section of the user interface.

        This function calls helper functions to display a description and set up
        a file uploader for the description
    """
    display_description()
    description_file_uploader()
    add_vertical_space(4)

def run_section():
    """
        Set up and display the 'Run AutoGluon' section of the user interface.
    """
    st.markdown("""
           <h1 style='
               font-weight: light;
               padding-left: 20px;
               padding-right: 20px;
               margin-left:60px;
               font-size: 2em;
           '>
               Run Autogluon
           </h1>
       """, unsafe_allow_html=True)
    col1, col2, col3,col4,col5= st.columns([1,10.9, 0.2, 10.9,1],gap='large')
    with col2:
        col11, col12 = st.columns(2)
        with col11:
            config_autogluon_preset()
            config_time_limit()
        with col12:
            config_transformer()
            config_llm()
        set_description()
    with col3:
        st.html(
            '''
                <div class="divider-vertical-line"></div>
                <style>
                    .divider-vertical-line {
                        border-left: 2px solid rgba(49, 51, 63, 0.2);
                        height: 590px;
                        margin: auto;
                    }
                </style>
            '''
        )
    with col4:
        file_uploader()
    update_config_overrides()
    _, mid_pos,_ = st.columns([1,22,1],gap='large')
    with mid_pos:
        run_button()
        if st.session_state.task_running:
            show_cancel_task_button()
    _, mid_pos, _ = st.columns([1, 22, 1], gap='large')
    with mid_pos:
        if st.session_state.task_running:
            show_real_time_logs()
        elif not st.session_state.task_running and not st.session_state.task_canceled:
            show_logs()
        elif st.session_state.task_canceled:
            show_cancel_container()
    generate_output_file()
    _, download_pos,_ = st.columns([5,2,5])
    with download_pos:
        download_button()
    st.markdown("---", unsafe_allow_html=True)

def main():
    get_user_session_id()
    run_section()


if __name__ == "__main__":
    main()