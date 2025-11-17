import requests
import sys

# data for a single student is contained in the dictionary below.  Modify individual values (keep as floats) to experiment with shifts in predicted outcomes
student = {
    'marital_status': 1.0,
    'app_mode': 39.0,
    'app_order': 1.0,
    'course': 9119.0,
    'day_evening': 1.0,
    'prev_qual': 1.0,
    'prev_qual_grade': 130.0,
    'nationality': 1.0,
    'mother_qual': 19.0,
    'father_qual': 37.0,
    'mother_occupation': 9.0,
    'father_occupation': 9.0,
    'admission_grade': 134.0,
    'displaced': 0.0,
    'special_needs': 0.0,
    'debtor': 0.0,
    'tuition_updated': 1.0,
    'gender': 1.0,
    'scholarship': 1.0,
    'enrolled_age': 24.0,
    'international': 0.0,
    'sem1_units_credited': 0.0,
    'sem1_units_enrolled': 5.0,
    'sem1_unit_evals': 8.0,
    'sem1_unit_approved': 5.0,
    'sem1_grades': 11.833333333333334,
    'sem1_units_noeval': 0.0,
    'sem2_units_credited': 0.0,
    'sem2_units_enrolled': 5.0,
    'sem2_unit_evals': 8.0,
    'sem2_unit_approved': 0.0,
    'sem2_grades': 9.0,
    'sem2_units_noeval': 0.0,
    'unemployment': 12.7,
    'inflation': 3.7,
    'gdp': -1.7
}

if len(sys.argv) > 1:
    # user-specified hosting target for webapp
    if sys.argv[1] == '1':
        print('PythonAnywhere-hosted solution targeted.  Executing...')
        url = "https://amackenzie.pythonanywhere.com/predict"
    elif sys.argv[1] == '2':
        print('Locally-hosted solution targeted.  Attempting to execute...')
        url = 'http://127.0.0.1:9696/predict'
else:
    print('No arguments provided.  Using PythonAnywhere-hosted solution as target.  Attempting to execute...')
    url = 'https://amackenzie.pythonanywhere.com/predict'
print(requests.post(url, json = student).json())

