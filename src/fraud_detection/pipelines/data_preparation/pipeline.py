from kedro.pipeline import Pipeline, node, pipeline
from .nodes import load_data, clean_data, feature_engineering

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=load_data,
            inputs="fraud_data",
            outputs="loaded_data",
            name="load_data_node"
        ),
        node(
            func=clean_data,
            inputs="loaded_data",
            outputs="cleaned_data",
            name="clean_data_node"
        ),
        node(
            func=feature_engineering,
            inputs="cleaned_data",
            outputs="features_data",
            name="feature_engineering_node"
        )
    ])
