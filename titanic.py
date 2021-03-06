import joblib
from datetime import datetime
import pandas
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin


class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, df: pandas.DataFrame):
        dfc = df.copy(deep=True)

        title_lookup = {"Mr.": "Mr", "Mrs.": "Mrs", "Miss.": "Miss", "Master.": "Master", "Dr.": "Dr"}

        for key in title_lookup.keys():
            title = title_lookup[key]

            dfc.loc[dfc["Name"].str.contains(key, regex=False), "Title"] = title

        return dfc


train_path = "train.csv"
test_path = "test.csv"
predict_path = "prediction_%Y%m%d_%H%M%S.csv"
model_path = "model_%Y%m%d_%H%M%S.pkl"
params_path = "params_%Y%m%d_%H%M%S.pkl"

timestamp = datetime.now()

train_df = pandas.read_csv(train_path, low_memory=False)
test_df = pandas.read_csv(test_path, low_memory=False)

target_column = 'Survived'

Xtrain = train_df.loc[:, train_df.columns != target_column]
Ytrain = train_df.loc[:, target_column].values
Xtest = test_df

categorical_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
categorical_transformer = make_pipeline(SimpleImputer(strategy='constant', fill_value='missing'),
                                        OneHotEncoder(handle_unknown='ignore'))

numeric_features = ['Age', 'Fare']
numeric_transformer = make_pipeline(SimpleImputer(strategy='median'))

preprocessor = Preprocessor()

column_transformer = ColumnTransformer(transformers=[('cat', categorical_transformer, categorical_features),
                                                     ('num', numeric_transformer, numeric_features)])

classifier = RandomForestClassifier(n_estimators=100)

pipeline = make_pipeline(preprocessor, column_transformer, classifier)

grid_params = {
    'randomforestclassifier__max_depth': [4, 8, 10, 12, 16],
    'randomforestclassifier__n_estimators': [50, 100, 200, 400]
}

grid = GridSearchCV(pipeline, grid_params, verbose=1, cv=5, n_jobs=2, refit=True)
grid.fit(Xtrain, Ytrain)

print("Best score: %f" % grid.best_score_)

best_pipeline = grid.best_estimator_
best_params = grid.best_params_
joblib.dump(best_pipeline, timestamp.strftime(model_path), compress=1)
joblib.dump(best_params, timestamp.strftime(params_path), compress=1)

test_df[target_column] = best_pipeline.predict(test_df)
test_df.to_csv(timestamp.strftime(predict_path), index=False)

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
