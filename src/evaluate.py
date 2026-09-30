import mlflow
import mlflow.sklearn

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
)

from src.data import (
    load_wine_data,
    split_data,
)


MODEL_URI = "models:/WineClassifier@champion"


def main():
    print("=" * 60)
    print("Wine Classifier - Champion Model Evaluation")
    print("=" * 60)

    print("\nLoading Wine dataset...")

    X, y = load_wine_data()

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
    )

    print(
        f"Training samples: {X_train.shape[0]}"
    )

    print(
        f"Testing samples: {X_test.shape[0]}"
    )

    print(
        "\nLoading registered model:"
    )

    print(
        MODEL_URI
    )

    model = mlflow.sklearn.load_model(
        MODEL_URI
    )

    print(
        "Champion model loaded successfully."
    )

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
    )

    loss = log_loss(
        y_test,
        probabilities,
    )

    print("\n" + "=" * 60)
    print("FINAL TEST SET RESULTS")
    print("=" * 60)

    print(
        f"Test Accuracy: {accuracy:.4f}"
    )

    print(
        f"Test Macro F1: {macro_f1:.4f}"
    )

    print(
        f"Test Log Loss: {loss:.4f}"
    )

    print("=" * 60)

    print(
        "\nInference verification completed successfully."
    )


if __name__ == "__main__":
    main()
