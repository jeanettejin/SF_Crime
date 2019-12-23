import pandas as pd
import pickle

from sklearn.pipeline import FeatureUnion

from Modeling.helpers import process_train, make_submission_file
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from Modeling.Transformers_Model import ColumnSelector, TypeSelector, MakeFeatures
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


from sklearn.metrics import make_scorer
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

from skopt.space.space import Real, Integer, Categorical
# https://github.com/YannDubs/scikit-optimize/blob/master/skopt/searchcv.py
from skopt import BayesSearchCV


# load in data
train = pd.read_csv('Data/train.csv')
test = pd.read_csv('Data/test.csv')

# process training set
train = process_train(train)

# train test split
X_train, X_val, y_train, y_val = train_test_split(train.loc[:, ~train.columns.isin(['Category'])],
                                                    train.loc[:, train.columns.isin(['Category'])],
                                                test_size=0.15, random_state=1, stratify=train['Category'])
print(len(y_train.Category.unique()))
print(len(y_val.Category.unique()))

# getting data and making features
feature_pipeline = Pipeline([('SelectCols', ColumnSelector(columns = ['Dates', 'PdDistrict', 'Address', 'X', 'Y'])),
                               ('MakeFeatures', MakeFeatures())])

# categorical pipeline
categorical_pipeline = Pipeline([('select_categorical', TypeSelector(dtype = 'object')),
                        ('dummy', OneHotEncoder(sparse=False, handle_unknown='ignore'))])

# numerical pipeline
numeric_pipeline = Pipeline([('select_numeric', TypeSelector(dtype = 'number'))])

# processing pipeline
cat_num_featun = FeatureUnion([('categorical', categorical_pipeline),
             ('numerical', numeric_pipeline)])


# combined pipeline
estimator_pipeline = Pipeline([('Features', feature_pipeline),
          ('Categorical_Numeric', cat_num_featun),
            ('Estimator', RandomForestClassifier())])


# search space
search_space = {
    "Estimator__n_estimators": Integer(35, 45),
    "Estimator__min_samples_split": Integer(95, 105),
}

# scorer
metric = make_scorer(score_func=log_loss,
                     greater_is_better=False,
                     needs_proba = True,
                     labels = train['Category'].unique())

# cv
kfold_cv = KFold(n_splits = 10 , shuffle=True, random_state=42)

# bayessearch cv
bayes_tuned_pipeline = BayesSearchCV(
    estimator=estimator_pipeline,
    search_spaces=search_space,
    n_iter=5,
    scoring=metric,
    cv=kfold_cv,
    verbose=12,
    n_jobs=-1,
    refit=True
)

bayes_tuned_pipeline.fit(X_train, y_train)

# # Load the model
# bayes_tuned_pipeline = pickle.load(open('tuned_pipeline.pkl','rb'))

# log loss
phat_val = bayes_tuned_pipeline.predict_proba(X_val)
log_loss(y_val, phat_val)

# submission file
make_submission_file(bayes_tuned_pipeline, test, 'onsite_s2.csv')





