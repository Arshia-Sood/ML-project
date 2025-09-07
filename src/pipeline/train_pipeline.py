from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    dt = DataTransformation()
    train_arr, test_arr, preprocessor_path = dt.initiate_data_transformation(
        "notebook/data/train.csv",
        "notebook/data/test.csv"
    )

    mt = ModelTrainer()
    score = mt.initiate_model_trainer(train_arr, test_arr, preprocessor_path)

    print("Model trained and saved at artifacts/model.pkl")
    print("R2 Score:", score)
