
import pandas as pd
import numpy as np

train = pd.read_csv('Data/train.csv')


def make_submission_file(model_pipeline, test, submission_name):
    """
    Function that writes out in submission format
    :param model_pipeline: estimator with predict proba method
    :param test: testing data
    :param submission_name: (str) name of file to write out to
    :return: (None) file is written out
    """
    test_ID = test["Id"]
    yhat_proba = model_pipeline.predict_proba(test)

    submission = pd.DataFrame(np.c_[test_ID, yhat_proba], columns=["Id"] + model_pipeline.classes_.tolist())
    submission["Id"] = submission["Id"].astype(int)
    submission.to_csv(submission_name, index=False)




def process_train(X):
    """
    Processing function that removes duplicates and rows with bad coordinates
    :param X:
    :return: train modified
    """
    X = X.drop_duplicates()
    X = X[X.Y < 40]
    return X

