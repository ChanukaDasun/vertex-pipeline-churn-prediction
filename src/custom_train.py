from google.cloud import aiplatform
import os
from dotenv import load_dotenv
from src.preprocess_dataset import preprocess_data
from src.model_init import get_model_config
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")
BUCKET_NAME = os.getenv("BUCKET_NAME")

def customer_churn_pred_train():

    aiplatform.init(project=PROJECT_ID, location=REGION)

    try:
        # Create a tabular dataset
        dataset = aiplatform.TabularDataset.create(
            display_name="churn-dataset",
            gcs_source=f"gs://{BUCKET_NAME}/data/raw/customer_churn_dataset.csv",
        )

        print(f"Dataset created: {dataset.resource_name}")
        path = f"gs://{BUCKET_NAME}/data/raw/customer_churn_dataset.csv"
        X_train, X_test, y_train, y_test = preprocess_data(dataset_path=path)
        model = get_model_config()

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("Model Evaluation Metrics:\n")
        print(f"Accuracy : {accuracy:.4f}\n")
        print(f"Precision: {precision:.4f}\n")
        print(f"Recall   : {recall:.4f}\n")
        print(f"F1-score : {f1:.4f}\n")

        return model

    except Exception as e:
        print(f"Error creating dataset: {e}")
        return

    return dataset, model

if __name__=="__main__":
    customer_churn_pred_train()