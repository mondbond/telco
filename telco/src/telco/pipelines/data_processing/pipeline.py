from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    telco_preprocess
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=telco_preprocess,
                inputs="telco",
                outputs=["telco_processed_x", "telco_processed_y"],
                name="process_telco_data_node",
            )
        ]
    )
