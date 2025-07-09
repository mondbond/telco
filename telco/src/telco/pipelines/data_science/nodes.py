from pathlib import Path

import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import mlflow.keras
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score

def split_telco_data(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    return X_train, X_val, y_train, y_val

def train_telco_nn_model(X_train, X_val, y_train, y_val, params):
    model = Sequential([
        Dense(32, input_shape=(X_train.shape[1],), activation="relu"),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer=Adam(learning_rate=params["lr"]), loss="binary_crossentropy",metrics=["accuracy"])

    with mlflow.start_run():
        mlflow.log_params(params)

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            verbose=0
        )

        val_accuracy = history.history["val_accuracy"][-1]
        mlflow.log_metric("val_accuracy", val_accuracy)

        mlflow.keras.log_model(model, artifact_path='telco_model')

    return model

def evaluate_telco_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred = prediciton_to_target(y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1_score = f1_score(y_test, y_pred)
    
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_f1_score", f1_score)
    mlflow.log_metric("test_recall", recall)

    confusion = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(confusion, annot=True, fmt="d", ax=ax)

    confusion_save_path = (Path(__file__).resolve().parents[4] / "data" / "08_reporting" / "conf_matrix.png")

    fig.savefig(confusion_save_path)
    mlflow.log_artifact(str(confusion_save_path))

def prediciton_to_target(y_pred, threshold=0.5):
    return (y_pred > threshold).astype(int)
