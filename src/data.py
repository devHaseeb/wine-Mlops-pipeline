from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


def load_wine_data():
    """
    Load the Wine dataset and perform basic validation.

    Returns:
        X: Feature data
        y: Target labels
    """

    wine = load_wine()

    X = wine.data
    y = wine.target

    # Check feature count
    if X.shape[1] != 13:
        raise ValueError(
            f"Expected 13 features, but found {X.shape[1]}"
        )

    # Check for null values
    if hasattr(X, "isnull"):
        if X.isnull().any().any():
            raise ValueError(
                "Feature data contains null values."
            )
    else:
        if X is None:
            raise ValueError(
                "Feature data is None."
            )

    if hasattr(y, "isnull"):
        if y.isnull().any():
            raise ValueError(
                "Target data contains null values."
            )
    else:
        if y is None:
            raise ValueError(
                "Target data is None."
            )

    return X, y


def split_data(X, y):
    """
    Perform a stratified 80/20 train-test split.

    Args:
        X: Feature data
        y: Target labels

    Returns:
        X_train, X_test, y_train, y_test
    """

    return train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )
