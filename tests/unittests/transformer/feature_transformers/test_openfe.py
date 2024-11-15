from autogluon.transformer import OpenFETransformer
from sklearn import datasets
from sklearn.model_selection import train_test_split


def test_openfe_transformer():
    ofet = OpenFETransformer(n_jobs=2)

    diabetes = datasets.load_iris(as_frame=True)
    X = diabetes.data
    y = diabetes.target

    input_columns = X.columns

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

    ofet._fit_dataframes(train_X, train_y)
    transformed_train_X, transformed_test_X = ofet._transform_dataframes(train_X, test_X)

    assert set(input_columns).issubset(set(transformed_train_X.columns))
    assert set(input_columns).issubset(set(transformed_test_X.columns))
