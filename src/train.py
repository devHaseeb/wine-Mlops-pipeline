import mlflow
import mlflow.sklearn

from mlflow.models import infer_signature

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
)

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
)

from src.data import (
    load_wine_data,
    split_data,
)


EXPERIMENT_NAME = "Wine-Cultivar-Classification"
MODEL_NAME = "WineClassifier"


RF_CONFIGS = [
    {
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42,
    },
    {
        "n_estimators": 200,
        "max_depth": 10,
        "random_state": 42,
    },
    {
        "n_estimators": 300,
        "max_depth": None,
        "random_state": 42,
    },
]


GB_CONFIGS = [
    {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 2,
        "random_state": 42,
    },
    {
        "n_estimators": 150,
        "learning_rate": 0.10,
        "max_depth": 3,
        "random_state": 42,
    },
    {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 3,
        "random_state": 42,
    },
]


def calculate_metrics(y_true, predictions, probabilities):
    """Calculate accuracy, macro F1 and log loss."""

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    macro_f1 = f1_score(
        y_true,
        predictions,
        average="macro",
    )

    loss = log_loss(
        y_true,
        probabilities,
    )

    return accuracy, macro_f1, loss


def evaluate_model(model, X_train, y_train):
    """Calculate train and 5-fold validation metrics."""

    # ---------------------------------------------------------
    # Train metrics
    # ---------------------------------------------------------

    model.fit(
        X_train,
        y_train,
    )

    train_predictions = model.predict(
        X_train
    )

    train_probabilities = model.predict_proba(
        X_train
    )

    train_accuracy, train_macro_f1, train_log_loss = (
        calculate_metrics(
            y_train,
            train_predictions,
            train_probabilities,
        )
    )

    # ---------------------------------------------------------
    # Validation metrics using 5-fold stratified CV
    # ---------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    validation_predictions = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict",
    )

    validation_probabilities = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
    )

    (
        validation_accuracy,
        validation_macro_f1,
        validation_log_loss,
    ) = calculate_metrics(
        y_train,
        validation_predictions,
        validation_probabilities,
    )

    return {
        "train_accuracy": train_accuracy,
        "train_macro_f1": train_macro_f1,
        "train_log_loss": train_log_loss,
        "validation_accuracy": validation_accuracy,
        "validation_macro_f1": validation_macro_f1,
        "validation_log_loss": validation_log_loss,
    }


def log_and_register_model(
    model,
    X_train,
    run_name,
    model_artifact_name,
):
    """Log model with signature and input example."""

    signature = infer_signature(
        X_train,
        model.predict(X_train),
    )

    input_example = X_train[:1]

    mlflow.sklearn.log_model(
        model,
        name=model_artifact_name,
        signature=signature,
        input_example=input_example,
        skops_trusted_types=[
            "sklearn.tree._tree.Tree"
        ],
    )


def train_random_forest(X_train, y_train):
    """Train and track all Random Forest configurations."""

    results = []

    for index, config in enumerate(
        RF_CONFIGS,
        start=1,
    ):
        run_name = f"RandomForest_Config_{index}"

        print(
            f"\nRunning {run_name}..."
        )

        model = RandomForestClassifier(
            **config
        )

        metrics = evaluate_model(
            model,
            X_train,
            y_train,
        )

        with mlflow.start_run(
            run_name=run_name
        ):

            mlflow.set_tag(
                "model_family",
                "RandomForest",
            )

            mlflow.set_tag(
                "configuration",
                f"RF_{index}",
            )

            mlflow.log_params(
                config
            )

            mlflow.log_metrics(
                metrics
            )

            log_and_register_model(
                model,
                X_train,
                run_name,
                "random_forest_model",
            )

            run_id = mlflow.active_run().info.run_id

        result = {
            "run_id": run_id,
            "run_name": run_name,
            "model_family": "RandomForest",
            "validation_macro_f1": metrics[
                "validation_macro_f1"
            ],
        }

        results.append(result)

        print(
            f"Train Accuracy: "
            f"{metrics['train_accuracy']:.4f}"
        )

        print(
            f"Train Macro F1: "
            f"{metrics['train_macro_f1']:.4f}"
        )

        print(
            f"Train Log Loss: "
            f"{metrics['train_log_loss']:.4f}"
        )

        print(
            f"Validation Accuracy: "
            f"{metrics['validation_accuracy']:.4f}"
        )

        print(
            f"Validation Macro F1: "
            f"{metrics['validation_macro_f1']:.4f}"
        )

        print(
            f"Validation Log Loss: "
            f"{metrics['validation_log_loss']:.4f}"
        )

    return results


def train_gradient_boosting(X_train, y_train):
    """Train and track all Gradient Boosting configurations."""

    results = []

    for index, config in enumerate(
        GB_CONFIGS,
        start=1,
    ):
        run_name = f"GradientBoosting_Config_{index}"

        print(
            f"\nRunning {run_name}..."
        )

        model = GradientBoostingClassifier(
            **config
        )

        metrics = evaluate_model(
            model,
            X_train,
            y_train,
        )

        with mlflow.start_run(
            run_name=run_name
        ):

            mlflow.set_tag(
                "model_family",
                "GradientBoosting",
            )

            mlflow.set_tag(
                "configuration",
                f"GB_{index}",
            )

            mlflow.log_params(
                config
            )

            mlflow.log_metrics(
                metrics
            )

            log_and_register_model(
                model,
                X_train,
                run_name,
                "gradient_boosting_model",
            )

            run_id = mlflow.active_run().info.run_id

        result = {
            "run_id": run_id,
            "run_name": run_name,
            "model_family": "GradientBoosting",
            "validation_macro_f1": metrics[
                "validation_macro_f1"
            ],
        }

        results.append(result)

        print(
            f"Train Accuracy: "
            f"{metrics['train_accuracy']:.4f}"
        )

        print(
            f"Train Macro F1: "
            f"{metrics['train_macro_f1']:.4f}"
        )

        print(
            f"Train Log Loss: "
            f"{metrics['train_log_loss']:.4f}"
        )

        print(
            f"Validation Accuracy: "
            f"{metrics['validation_accuracy']:.4f}"
        )

        print(
            f"Validation Macro F1: "
            f"{metrics['validation_macro_f1']:.4f}"
        )

        print(
            f"Validation Log Loss: "
            f"{metrics['validation_log_loss']:.4f}"
        )

    return results


def register_champion(all_results):
    """Register the best run and assign champion alias."""

    print("\n" + "=" * 60)
    print("CHAMPION MODEL SELECTION")
    print("=" * 60)

    champion = max(
        all_results,
        key=lambda result: result[
            "validation_macro_f1"
        ],
    )

    print(
        f"Champion Run: "
        f"{champion['run_name']}"
    )

    print(
        f"Validation Macro F1: "
        f"{champion['validation_macro_f1']:.4f}"
    )

    run_id = champion["run_id"]

    artifact_name = (
        "random_forest_model"
        if champion["model_family"] == "RandomForest"
        else "gradient_boosting_model"
    )

    model_uri = (
        f"runs:/{run_id}/{artifact_name}"
    )

    print(
        f"Model URI: {model_uri}"
    )

    print(
        "\nRegistering champion model..."
    )

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
    )

    version = registered_model.version

    print(
        f"Registered Model: {MODEL_NAME}"
    )

    print(
        f"Model Version: {version}"
    )

    client = mlflow.MlflowClient()

    client.set_registered_model_alias(
        MODEL_NAME,
        "champion",
        version,
    )

    print(
        f"Alias assigned: champion -> version {version}"
    )

    return champion


def main():
    print("=" * 60)
    print("Wine MLOps Training Pipeline")
    print("=" * 60)

    print("\nLoading Wine dataset...")

    X, y = load_wine_data()

    print(
        f"Samples: {X.shape[0]}"
    )

    print(
        f"Features: {X.shape[1]}"
    )

    print(
        "\nPerforming 80/20 stratified split..."
    )

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

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    print("\n" + "=" * 60)
    print("RANDOM FOREST")
    print("=" * 60)

    rf_results = train_random_forest(
        X_train,
        y_train,
    )

    print("\n" + "=" * 60)
    print("GRADIENT BOOSTING")
    print("=" * 60)

    gb_results = train_gradient_boosting(
        X_train,
        y_train,
    )

    all_results = (
        rf_results + gb_results
    )

    champion = register_champion(
        all_results
    )

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    for result in all_results:
        print(
            f"{result['run_name']}: "
            f"Validation Macro F1 = "
            f"{result['validation_macro_f1']:.4f}"
        )

    print("\n" + "=" * 60)

    print(
        f"Champion: {champion['run_name']}"
    )

    print(
        f"Champion Validation Macro F1: "
        f"{champion['validation_macro_f1']:.4f}"
    )

    print(
        f"Registered Model: {MODEL_NAME}"
    )

    print(
        "Alias: champion"
    )

    print("=" * 60)

    print(
        "\nTraining and model registration "
        "completed successfully."
    )


if __name__ == "__main__":
    main()
