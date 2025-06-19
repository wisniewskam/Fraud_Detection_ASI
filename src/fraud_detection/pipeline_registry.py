from fraud_detection.pipelines import data_preparation, model_training

def register_pipelines():
    return {
        "data_preparation": data_preparation.create_pipeline(),
        "model_training": model_training.create_pipeline(),
        "__default__": data_preparation.create_pipeline() + model_training.create_pipeline(),
    }
