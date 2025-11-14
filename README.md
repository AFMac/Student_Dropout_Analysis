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


