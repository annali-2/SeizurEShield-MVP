# SeizurEShield MVP Flask Application

This repository contains a Flask web application for uploading EEG files, processing them, running predictions using a pre-trained RNN-LSTM model, and visualizing the results.

### Features
* Upload EEG files in .edf format
* Process uploaded files and extract EEG channels and duration
* Run predictions using a pre-trained RNN-LSTM model
* Visualize the processed EEG data

## Setup Instructions
1. Clone the Repository

    ```
    git clone https://github.com/annali-2/seizure.git
    cd seizure
    ```

2. Create a Virtual Environment
It's recommended to create a virtual environment to manage dependencies.

    ```
    python -m venv venv
    ```

3. Activate the Virtual Environment
    
    On Windows:

    ```
    venv\Scripts\activate
    ```

    On macOS/Linux:

    ```
    source venv/bin/activate
    ```

4. Install Dependencies

    ``` 
    pip install -r requirements.txt
    ```

5. Ensure Necessary Directories Exist

    Make sure the following directories exist in the project root:
    * uploads/
    * data/
    * model_dir/


6. Run the Application

    ```
    flask --app app run
    ```

8. Access the Application

    Open your web browser and go to http://127.0.0.1:5000/ to access the application.

### Usage
* Upload EEG File: Navigate to the upload page and upload a .edf file.
* Process and Analyze: After uploading, the app processes the file and displays the channels and duration.
* Run Predictions: Select a montage and run predictions on the processed data.
* Download Results: Download the processed file with predictions.
* Visualize Data: View the visual representation of the EEG data.



