# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Description

This is a Flask application that visualizes the distribution of message categories using Plotly. The application uses a machine learning model to classify messages into different categories, and then displays the distribution of these categories in a series of interactive plots.

## Installation

1. Clone the repository:
    ```
    git clone https://github.com/edgtzolmedo/udacity-disaster-response-pipeline.git
    ```

2. Navigate to the project directory:
    ```
    cd app
    ```

3. Install the required packages:
    ```
    pip install -r requirements.txt
    ```

## Features

- **Message Classification**: The application uses a machine learning model to classify messages into different categories.

- **Interactive Visualizations**: The application displays the distribution of message categories in a series of interactive plots created with Plotly.

## License

This project is licensed under the terms of the [MIT License](LICENSE).