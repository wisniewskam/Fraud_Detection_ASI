from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_model, analyze_feature_importance

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_model,
            inputs=dict(data="features_data", label_column="params:label_column"),
            outputs="predictor_path",
            name="train_model_node"
        ),
        node(
            func=analyze_feature_importance,
            inputs=dict(predictor_path="predictor_path", data="features_data"),
            outputs="feature_importance",
            name="feature_importance_node"
        )
    ])
