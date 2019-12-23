from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        """
        :param columns: List of column names in X to select
        """
        self.columns = columns
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        :param X: pd.DataFrame
        :return: pd.DataFrame of selected Columns
        """
        assert isinstance(X, pd.DataFrame)

        try:
            return X.loc[:, self.columns]

        except KeyError:
            unknown_columns = list(set(self.columns) - set(X.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % unknown_columns)


class TypeSelector(BaseEstimator, TransformerMixin):
    """
    Transformer that returns dataframe of specified dtype
    """
    def __init__(self, dtype=None):
        """
        :param dtype: one of 'object', 'bool' or 'number'
        """
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include=[self.dtype])


class MakeFeatures(BaseEstimator, TransformerMixin):
    """
    Transformer that makes features used in final model
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        # from date
        X['Dates'] = pd.to_datetime(X['Dates'])
        X['N_Days'] = (X['Dates'] - X['Dates'].min()).dt.days
        X['Day'] = X['Dates'].dt.day
        X['DayOfWeek'] = X['Dates'].dt.weekday
        X['Month'] = X['Dates'].dt.month
        X['Year'] = X['Dates'].dt.year
        X['Hour'] = X['Dates'].dt.hour
        X['Minute'] = X['Dates'].dt.minute

        # from address
        X['Block'] = X['Address'].str.contains('block', case=False).astype(int)

        X = X.drop(columns=['Dates', 'Address'])

        return X