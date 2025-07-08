from typing import Dict

from kedro.pipeline import Pipeline


from telco.pipelines.data_processing import pipeline as data_processing_pipeline
from telco.pipelines.data_science import pipeline as model_training_pipeline

def register_pipelines() -> Dict[str, Pipeline]:

    return {
        "dp": data_processing_pipeline.create_pipeline(),
        "train": model_training_pipeline.create_pipeline(),

        "__default__": (
                data_processing_pipeline.create_pipeline()
                + model_training_pipeline.create_pipeline()
        )
    }