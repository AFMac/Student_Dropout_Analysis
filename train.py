#key library imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import pickle


# set parameters
xgb_params = {
        'eta': 0.05,
        'max_depth': 6,
        'min_child_weight': 1,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'nthread': 4,
        'seed': 1,
        'verbosity': 1,
    }
n_splits = 10
output_file = 'xgboost_model.bin'

# data preparation
df = pd.read_csv('data.csv', sep = ';', header=0)
cols = ['marital_status', 'app_mode', 'app_order', 'course', 'day_evening', 'prev_qual', 'prev_qual_grade', 'nationality', 'mother_qual',
        'father_qual', 'mother_occupation', 'father_occupation', 'admission_grade', 'displaced', 'special_needs', 'debtor', 'tuition_updated',
        'gender', 'scholarship', 'enrolled_age', 'international', 'sem1_units_credited', 'sem1_units_enrolled', 'sem1_unit_evals', 'sem1_unit_approved',
        'sem1_grades', 'sem1_units_noeval', 'sem2_units_credited', 'sem2_units_enrolled', 'sem2_unit_evals', 'sem2_unit_approved', 'sem2_grades',
        'sem2_units_noeval', 'unemployment', 'inflation', 'gdp', 'target']
df.columns = cols

# dataset currently has target encoded as a 3-way classification.  since we haven't addressed this yet via the class, adding a separate column
# to turn this into a binary classification problem (either dropped out or not).  Stretch goal will be to evaluate both binary and multi-state classifications...
target_trans = {'Graduate': 0, 'Dropout': 1, 'Enrolled': 0}
df['dropout'] = df['target'].apply(lambda x: target_trans[x])
df.dropout.value_counts()

# since the target column is the sole categorical column, shifting that to binary for ease of analysis
target_trans = {'Graduate': 0, 'Dropout': 1, 'Enrolled': 2}
df['target'] = df['target'].apply(lambda x: target_trans[x])


df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state = 11)
y_full_train = df_full_train.dropout.values
y_test = df_test.dropout.values

#train the model

def train(df_train, y_train):
    dicts = df_train.to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names = list(dv.get_feature_names_out()))
    model = xgb.train(xgb_params, dtrain, num_boost_round = 150)

    return dv, model

def predict(df_target, dv, model):
    dict = df_target.to_dict(orient = 'records')
    X_test = dv.transform(dict)

    d_test = xgb.DMatrix(X_test, feature_names = list(dv.get_feature_names_out()))
    return model.predict(d_test)

dv, model = train(df_full_train.drop(['dropout', 'target'], axis=1), y_full_train)
y_pred = predict(df_test.drop(['dropout', 'target'], axis=1), dv, model)
print('Model initial AUC score = %.3f' % roc_auc_score(y_test, y_pred))

# model validation
print(f'Validating model across {n_splits} KFold splits')
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=11)

scores = []
fold = 0

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.dropout.values
    y_val = df_val.dropout.values

    dv, model = train(df_train.drop(['dropout', 'target'], axis=1), y_train)
    y_pred = predict(df_val.drop(['dropout', 'target'], axis=1), dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)
    print(f'AUC on fold {(fold+1)} is {auc}')
    fold += 1

print('Validation results:')
print('AUC performance across %s folds (average +/- std dev) = %.3f +- %.3f' % (fold, np.mean(scores), np.std(scores)))

# save the model
with open(output_file, 'wb') as f_out:  # 'wb' means write-binary
    pickle.dump((dv, model), f_out)

print(f'Model saved to {output_file}')