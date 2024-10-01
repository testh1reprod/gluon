import streamlit as st
import os
import pandas as pd
import uuid
import glob
import subprocess
import psutil
import streamlit.components.v1 as components
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.add_vertical_space import add_vertical_space



CONFIG_DIR = '../../../config'
BASE_DATA_DIR = './user_data'

os.makedirs(BASE_DATA_DIR, exist_ok=True)
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
    if 'user_session_id' not in st.session_state:
        st.session_state.user_session_id = str(uuid.uuid4())
    return st.session_state.user_session_id


def generate_output_filename():
    user_session_id = get_user_session_id()
    unique_id = user_session_id[:8]
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f"output_{unique_id}_{timestamp}.csv"
    return output_filename

def get_user_data_dir():
    # Generate a unique directory name for the user session if it doesn't exist
    if 'user_data_dir' not in st.session_state:
        unique_dir = st.session_state.user_session_id
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


# Run the autogluon-assistant command
def run_autogluon_assistant(config_dir, data_dir):
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
    if st.session_state.output_file is not None and st.session_state.task_running is False:
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

def test_uploader():
    test_file = st.file_uploader("Upload Test Dataset", key = 'test_file_uploader',label_visibility="collapsed")
    user_data_dir = get_user_data_dir()
    if test_file is not None:
        save_uploaded_file(test_file, os.path.join(user_data_dir, "test.csv"))
        test_df = pd.read_csv(test_file)
        st.session_state.test_file_name = test_file.name
        st.session_state.test_file_df = test_df

def sample_output_uploader():
    sample_output_file = st.file_uploader("Upload Sample Output Dataset (Optional)", key="sample_output_file_uploader",label_visibility="collapsed")
    user_data_dir = get_user_data_dir()
    if sample_output_file is not None:
        save_uploaded_file(sample_output_file, os.path.join(user_data_dir, "sample_output.csv"))
        sample_output_df = pd.read_csv(sample_output_file)
        st.session_state.sample_output_file_name = sample_output_file.name
        st.session_state.sample_output_file_df = sample_output_df

def file_uploader():
    with stylable_container(key='train_file_uploader',css_styles="""
        {
            border: 1px solid rgba(49, 51, 63, 0.2);
            padding: calc(1em - 1px);
            background-color:transparent;
            display: flex;
            border-radius: 10px;
            flex-direction: column;
            align-items: center;
        }
    """):
        st.html(
            """
             <div style="display: flex; align-items: center; justify-content: center; height: 80%;">
            <i class="fa-brands fa-buromobelexperte" style="font-size: 40px; margin-top: 30px;color: #1590e1;"></i>
            </div>
            """
        )
        train_uploader()
    if st.session_state.train_file_name is not None:
        with st.popover(st.session_state.train_file_name,use_container_width=True):
            st.write(st.session_state.train_file_df.head(10))

    with stylable_container(key='test_file_uploader', css_styles="""
               {
                   border: 1px solid rgba(49, 51, 63, 0.2);
                   padding: calc(1em - 1px);
                   background-color: transparent;
                   display: flex;
                    border-radius: 10px;
                   flex-direction: column;
                   align-items: center;
               }
           """):
        st.html(
            """
             <div style="display: flex; align-items: center; justify-content: center; height: 80%;">
            <i class="fa-solid fa-gear" style="font-size: 40px; margin-top: 30px;color: #1590e1;"></i>
            </div>
            """
        )
        test_uploader()
    if st.session_state.test_file_name is not None:
        with st.popover(st.session_state.test_file_name, use_container_width=True):
            st.write(st.session_state.test_file_df.head(10))

    with stylable_container(key='sample_output_file_uploader', css_styles="""
                {
                    border: 1px solid rgba(49, 51, 63, 0.2);
                    padding: calc(1em - 1px);
                    background-color: transparent;
                    display: flex;
                    border-radius: 10px;
                    flex-direction: column;
                    align-items: center;
                }
            """):
        st.html(
            """
             <div style="display: flex; align-items: center; justify-content: center; height: 80%;">
            <i class="fa-solid fa-file-csv" style="font-size: 40px; margin-top: 30px;color: #1590e1;"></i>
            </div>
            """
        )
        sample_output_uploader()
    if st.session_state.sample_output_file_name is not None:
        with st.popover(st.session_state.sample_output_file_name, use_container_width=True):
            st.write(st.session_state.sample_output_file_df.head(10))
    add_vertical_space(3)


def toggle_running_state():
    st.session_state.task_running = True
    st.session_state.task_canceled = False

def toggle_cancel_state():
    st.session_state.task_canceled = True

def run_button():
    user_data_dir = get_user_data_dir()
    if st.button(label="🔘&nbsp;&nbsp;Run!", on_click=toggle_running_state,
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

def set_description():
    display_description()
    description_file_uploader()
    add_vertical_space(4)

def run_section():
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