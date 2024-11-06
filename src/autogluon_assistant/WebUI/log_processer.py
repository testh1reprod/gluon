import re
import time

import streamlit as st
from constants import TIME_LIMIT_MAPPING
from stqdm import stqdm


def parse_model_path(log):
    """
    Extract the AutogluonModels path from the log text.

    Args:
        log (str): The log text containing the model path

    Returns:
        str or None: The extracted model path or None if not found
    """
    pattern = r'"([^"]*AutogluonModels[^"]*)"'
    match = re.search(pattern, log)
    if match:
        return match.group(1)
    return None


def show_log_line(line):
    """
    Show log line based on prefix and Rich syntax
    - Lines starting with WARNING: → st.warning
    - Other lines → st.markdown
    """
    if "INFO:" in line:
        line = line.split(":", 1)[1].split(":", 1)[1]
    if any(
        message in line
        for message in [
            "Task understanding complete",
            "Automatic feature generation complete",
            "Model training complete",
            "Prediction complete",
        ]
    ):
        return st.success(line)
    elif line.startswith("WARNING:"):
        return st.warning(line)
    return st.markdown(line)


def get_stage_from_log(log_line):
    if "Task understanding starts" in log_line:
        return "Task Understanding"
    elif "Automatic feature generation starts" in log_line:
        return "Feature Generation"
    elif "Model training starts" in log_line:
        return "Model Training"
    elif "Prediction starts" in log_line:
        return "Prediction"
    return None


def show_logs():
    """
    Display logs and task status when task is finished.
    """
    if st.session_state.logs:
        tab1, tab2 = st.tabs(["Messages", "Logs"])
        with tab1:
            for stage, logs in st.session_state.stage_container.items():
                if logs:
                    with st.status(stage, expanded=False, state="complete"):
                        for log in logs:
                            show_log_line(log)
        with tab2:
            status_container = st.empty()
            log_container = st.empty()
            log_container.text_area("Real-Time Logs", st.session_state.logs, height=400)
            if st.session_state.return_code == 0:
                status_container.success("Task completed successfully!")
            else:
                status_container.error("Error detected in the process...Check the logs for more details")


def format_log_line(line):
    """
    Format log lines by removing ANSI escape codes, formatting markdown syntax,
    and cleaning up process-related information.

    Args:
        line (str): Raw log line to be formatted.

    Returns:
        str: Formatted log line with:
    """
    line = re.sub(r"\x1B\[1m(.*?)\x1B\[0m", r"**\1**", line)
    line = re.sub(r"^#", r"\\#", line)
    line = re.sub(r"\033\[\d+m", "", line)
    line = re.sub(r"^(\s*)\(_dystack pid=\d+\)\s*", r"\1", line)
    line = line.strip()
    return line


def process_realtime_logs(line):
    """
    Handles the real-time processing of log lines, updating the UI state,
    managing progress bars, and displaying status updates in the  interface.

    Args:  line (str): A single line from the log stream to process.
    """
    stage = get_stage_from_log(line)
    if stage:
        if stage != st.session_state.current_stage:
            st.session_state.current_stage = stage
        if stage not in st.session_state.stage_status:
            st.session_state.stage_status[stage] = st.status(stage, expanded=False)

    if st.session_state.current_stage:
        st.session_state.stage_status[st.session_state.current_stage].update(
            state="running",
        )
        if "AutoGluon training complete" in line:
            st.session_state.show_remaining_time = False
            st.session_state.stage_container[st.session_state.current_stage].append("st.session_state.progress_bar")
        with st.session_state.stage_status[st.session_state.current_stage]:
            time.sleep(1)
            if "Fitting model" in line and not st.session_state.show_remaining_time:
                st.session_state.progress_bar = stqdm(
                    desc="Elapsed Time for Fitting models: ", total=TIME_LIMIT_MAPPING[st.session_state.time_limit]
                )
                st.session_state.show_remaining_time = True
                st.session_state.increment_time = 0
            if st.session_state.show_remaining_time:
                st.session_state.increment_time += 1
                if st.session_state.increment_time <= TIME_LIMIT_MAPPING[st.session_state.time_limit]:
                    st.session_state.progress_bar.update(1)
            if not st.session_state.show_remaining_time:
                st.session_state.stage_container[st.session_state.current_stage].append(line)
                show_log_line(line)


def messages():
    """
    Handles the streaming of log messages from a subprocess, updates the UI with progress
    indicators, and manages the display of different training stages.
    processes ANSI escape codes, formats markdown, and updates various progress indicators.

    """
    if st.session_state.process is not None:
        process = st.session_state.process
        st.session_state.logs = ""
        progress = st.progress(0)
        task_stages = {
            "Task loaded!": 10,
            "Beginning AutoGluon training": 25,
            "Preprocessing data": 50,
            "User-specified model hyperparameters to be fit": 75,
            "Fitting model": 65,
            "AutoGluon training complete": 90,
        }
        status_container = st.empty()
        status_container.info("Running Tasks...")
        for line in process.stdout:
            print(line, end="")
            line = format_log_line(line)
            st.session_state.logs += line
            # if "exception" in line.lower():
            #     status_container.error("Error detected in the process...Check the logs for more details")
            #     st.session_state.output_file = None
            #     st.session_state.output_filename = None
            #     st.session_state.process = None
            #     st.session_state.pid = None
            #     st.session_state.task_running = False
            #     st.rerun()
            if "TabularPredictor saved" in line:
                model_path = parse_model_path(line)
                if model_path:
                    st.session_state.model_path = model_path
            if "Prediction complete" in line:
                status_container.success("Task completed successfully!")
                progress.progress(100)
            else:
                for stage, progress_value in task_stages.items():
                    if stage.lower() in line.lower():
                        progress.progress(progress_value / 100)
                        status_container.info(stage)
                        break
            process_realtime_logs(line)
