# Student_Dropout_Analysis
Midterm CAPSTONE project for MLZoomcamp 2025

# What is this?
This repository contains all elements required (data and Python scripts/Jupyter notebooks, Dockerfile, etc.) to meet requirements for the 2025 cohort of datatalk.club's [Machine Learning Zoomcamp](https://datatalks.club/blog/guide-to-free-online-courses-at-datatalks-club.html#machine-learning-zoomcamp).

# What is the problem?
This is a classification problem, focused on predicting student dropout and academic success, leveraging various socio-economic, demographic, and student academic factors over their first two semesters of undergraduate work.  Data is sourced from the [UC-Irvine Machine Learning Repository](https://archive-beta.ics.uci.edu/).  Specifically, the [Predict Students' Dropout and Academic Success](https://archive-beta.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) dataset.

Although formulated as a three-category classification problem, a new calculated feature (`dropout`) was added to turn this into a binary classification problem since the cohort hadn't yet addressed multi-class instances.

The dataset enables the training of a classification model using various datapoints (student academic background and progress through 2nd semester, socio-economic factors such as GDP and inflation, and demographics such as parent occupation/background, nationality, etc.).  Scripts establishing a web-service to serve the model allow for queries on individual students, returning assessed probability of dropout as well as a simple boolean "True/False" response (using `0.5` as the decision threshold).  Assumed use-case is for researchers or academic advisors to assess a student's likelihood of dropping out to inform the need for potential mitigation strategies.

# Repository Elements
- Data
  - Data is provided in the `data.csv` file.  The data itself has 36 features across 4,424 instances (each representing a single student).
  - Features are Real, Categorical, and Integer, however all but one of the categorical features were already label encoded, leaving a single non-numeric feature in the original dataset.
- Scripts
  - `notebook.ipynb`
    - Exploratory Data Analysis, data cleaning, initial model exploration and tuning
  - `finalized_models.ipynb`
    - Consolidated evaluation and comparison of final tuned models
  - `train.py`
    - Python script to train the final model and export to a binary file via `pickle`
  - `predict.py`
    - Python script to load the exported model and serve it via a `Flask` webservice
  - Containerized Execution
    - `Dockerfile_student_dropouts`, `requirements.txt`, `xgboost_model.bin` - Various elements (including exported model) to support execution via Docker container
  - Cloud Deployment
    - This project is currently hosted via [PythonAnywhere](https://www.pythonanywhere.com) and you can test it live.
    - `predict-test.py` can be run via your preferred Python environment.  The script sends a sample data request for a single student to the webapp and receives a response of the raw probability of their dropping out, and the accompanying prediction as 'True' or 'False' (based on a threshold of `p(dropout) >= 0.5`).
    - `predict.py` is included if you're interested in the code for the webapp itself.
   
# This is *boring*, I just want to see what this does.

No problem, all you'll need is the `predict-test.py` file included in this repository.  Execute it in your Python environment of choice, and you will receive a response that looks something like
> `{'dropout_decision': True, 'dropout_probability': 0.6174503244940356}`
>
> `Process finished with exit code 0`
The webapp is live on the PythonAnywhere platform, so nothing else special is required.

- What exactly is going on?
  - As you can see from looking at the straightforward `predict-test.py` script, when you execute it, you send data for a single student to a web service.  The individual data elements correspond to the socio-economic, demographic, and academic features mentioned above, and each has varying degrees of influence on the prediction, based on the model that has been trained on the data.  That trained model is used by the web app to generate a prediction based on the data you provide.  As mentioned, you can change the predicted outcome by modifying values in the data (the dictionary `student`) and re-running the script.  To execute:
    1. Download the file
    2. From that directory, run the file and provide '1' as an argument, as shown below
    ```python
    python predict_test.py 1 #passing the argument 1 will select the PythonAnywhere-hosted solution to send the data to.
    ```
    - Hint: if you'd like to see significant shifts, change the values for `sem2_unit_approved` and `sem2_grades`, as these features have a great deal of influence on model outcomes.  Feel free to experiment and change any other values as well.
    - The response you get is the calculated prediction of whether the student will drop out (`dropout_decision`) and the raw predicted probability (`dropout_probability`).
    - **Note**: this is a simplistic application with no error handling.  Be sure to maintain proper syntax (i.e. the dictionary must remain a dictionary, and values should be in the form of floats).
  - Of note, the free tier on PythonAnywhere was unable to accommodate the best performing model produced for this project (using the `XGBoost` library), so a simple linear regression model was used instead to allow for hosting with live demonstration capabilities.

***

# Data
- Source: [Predict Students' Dropout and Academic Success](https://archive-beta.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
- Details:
  - 4,424 datapoints
  - 36 features - integer and continuous encoded.  Many features reflect categorical data, but all were label encoded as integers.
  - No missing datapoints.
  - Target was categorical, reflecting `dropout`, `enrolled`, or `graduate`.  For this project, a new target of `dropout` was created, with values of `0` or `1`, with `target` values of `enrolled` and `graduate` combined to `0`, setting the stage for binary classification.
- Exploratory data analysis, and initial model training and tuning was performed and is available via the `notebook.ipynb` notebook.

# Model Selection
- Multiple models were evaluated for this project:
  - Linear regression
  - Decision Tree
  - Random Forest
  - XGBoost Classifier
- The linear regression and XGBoost models performed the best of the bunch, and were actually very comparable.  Ultimately, the XGBoost as the "best" and was selected for containerization and deployment.  However, due to limitations of the selected hosting platform, the linear regression model was ultimately utilized.
- Model training and tuning is contained in the `notebook.ipynb` file following exploratory data analysis.  Finalized models and some accompanying evaluation metrics and visualizations are captured in the `addtl_testing.ipynb` notebook.

# How to Install and Run
## Option 1: Local installation  (prerequisite: Python 3.13+)
1. Clone the repository
```python
git clone https://github.com/AFMac/Student_Dropout_Analysis.git
cd Student_Dropout_Analysis
```
2. Create a virtual environment
```python
python -m venv venv
venv\Scripts\activate    # on Linux: source venv/bin/activate
```
3. Install dependencies
```python
pip install -r requirements.txt
```
4. Train the model.  *This is optional - pre-trained models are available via .bin files and can be used without running this code.*
```python
python train.py
```
5. Start the local Flask API and begin serving your trained model.
```python
python predict.py
```

## Option 2: Containerized solution via Docker
1. Clone the repository or download the `Dockerfile_student_dropouts`, `requirements.txt`, `predict.py`, and `xgboost_model.bin` files and place in a working directory of your choice.  If you decided to clone the full repository, you can use the following commands:
```python
git clone https://github.com/AFMac/Student_Dropout_Analysis.git
cd Student_Dropout_Analysis
```
2. Build your Docker image.
```python
docker build -f Dockerfile_student_dropouts -t dropout_prediction .
```
3. Run your new container.
```python
docker run -it -p 9696:9696 dropout_prediction
```
4. Test your trained model.
```python
python predict_test.py 2      # providing '2' as an argument selects your local machine as the target to send the test data packet to.
```


 







