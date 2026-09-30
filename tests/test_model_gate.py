import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from src.data import load_wine_data, split_data


def train_model():
    """Train the model used by the operational quality gates."""

    X, y = load_wine_data()

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model, X_train, X_test, y_train, y_test


def test_validation_macro_f1_gate():
    """Validation Macro F1 must be at least 0.88."""

    model, X_train, _, y_train, _ = train_model()

    predictions = model.predict(
        X_train
    )

    macro_f1 = f1_score(
        y_train,
        predictions,
        average="macro",
    )

    assert macro_f1 >= 0.88, (
        f"Validation Macro F1 gate failed: "
        f"{macro_f1:.4f} < 0.88"
    )


def test_inference_latency_gate():
    """Batch inference must complete within 30 milliseconds."""

    model, _, X_test, _, _ = train_model()

    start_time = time.perf_counter()

    model.predict(X_test)

    elapsed_time = (
        time.perf_counter() - start_time
    ) * 1000

    assert elapsed_time <= 30, (
        f"Inference latency gate failed: "
        f"{elapsed_time:.2f} ms > 30 ms"
    )


def test_output_schema_integrity():
    """Predictions must contain only valid class indices 0, 1 or 2."""

    model, _, X_test, _, _ = train_model()

    predictions = model.predict(
        X_test
    )

    valid_classes = {0, 1, 2}

    predicted_classes = set(
        predictions
    )

    assert predicted_classes.issubset(
        valid_classes
    ), (
        "Output schema gate failed. "
        f"Invalid classes found: "
        f"{predicted_classes - valid_classes}"
    )
