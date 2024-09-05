# Autogluon Assistant Web UI

The Autogluon Assistant Web UI is a user-friendly application that allows users to leverage the capabilities of the Autogluon-Assistant library through an intuitive web interface.

## Features

1. **Dataset Upload and Task Definition**: Users can upload one or more CSV files as their dataset and provide a free-form text description of the task they want to solve.
2. **Data Preview**: The web UI allows users to preview the uploaded datasets, enabling them to inspect the data before initiating an Autogluon-Assistant run.
3. **Autogluon-Assistant Configuration**: Users can configure the Autogluon-Assistant run through an intuitive user interface, with the ability to customize the parameters defined in the YAML configuration file.
4. **Run Execution and Monitoring**: Users can initiate the Autogluon-Assistant run and track the progress of the pipeline execution, with the help of status updates and visual indicators like progress bars.
5. **Results Reporting**: After the run is complete, users can view the evaluation metrics, download the submission file (predictions), and optionally download the trained Autogluon model.
6. **Multi-Session Support**: The web UI supports multiple user sessions, ensuring that each user's actions and data are isolated and secure, enabling concurrent users.

## Installation

To install and run the Autogluon Assistant Web UI, follow these steps:

1. Clone the repository:
````
git clone https://github.com/tianyuanzoe/AGA-UI.git
````
2. Navigate to the project directory:
````
src/autogluon_assistant/Streamlit
````
3. Install the required dependencies:
````
pip install -r requirements.txt
````
4. Run the Streamlit app:
````
streamlit run app.py
````
5. The Autogluon Assistant Web UI should now be accessible in your web browser at `http://localhost:8501`
