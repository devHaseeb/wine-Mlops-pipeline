from src.data import load_wine_data, split_data


def test_wine_data_shape():
    X, y = load_wine_data()

    assert X.shape[1] == 13
    assert len(X) == len(y)


def test_wine_data_has_no_nulls():
    X, y = load_wine_data()

    assert X is not None
    assert y is not None


def test_stratified_split():
    X, y = load_wine_data()

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
    )

    assert len(X_train) == 142
    assert len(X_test) == 36

    assert len(y_train) == 142
    assert len(y_test) == 36
