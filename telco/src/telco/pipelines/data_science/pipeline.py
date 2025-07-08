from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_telco_data
from .nodes import train_telco_nn_model
from .nodes import evaluate_telco_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_telco_data,
                inputs=["telco_processed_x", "telco_processed_y"],
                outputs=["X_train", "X_val", "y_train", "y_val"],
                name="split_telco_data_node"
            ),
            node(
                func=train_telco_nn_model,
                inputs=["X_train", "X_val", "y_train", "y_val", "params:nn_params"],
                outputs="model_telco",
                name="train_telco_nn_model_node",
            ),
            node(
                func=evaluate_telco_model,
                inputs=["model_telco", "X_val", "y_val"],
                outputs=None,
                name="evaluate_telco_nn_model_node",
            ),
        ]
    )
