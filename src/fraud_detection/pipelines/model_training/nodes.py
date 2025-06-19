from autogluon.tabular import TabularPredictor
import pandas as pd

def train_model(data, label_column: str) -> str:
    predictor_path = "outputs/predictor"
    predictor = TabularPredictor(label=label_column, path=predictor_path).fit(data)
    return predictor_path

def analyze_feature_importance(predictor_path: str, data: pd.DataFrame) -> pd.DataFrame:
    predictor = TabularPredictor.load(predictor_path)
    importance_df = predictor.feature_importance(data)
    return importance_df
