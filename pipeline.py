from dotenv import load_dotenv
import os
from src.bucket_init import init_bucket_and_upload
from src.custom_train import customer_churn_pred_train
from src.deploy_model import deploy_model

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")
BUCKET_NAME = os.getenv("BUCKET_NAME")


def main_pipeline():
    """Main function for the pipeline."""
    try:
        storage_client, bucket = init_bucket_and_upload()
        model = customer_churn_pred_train()

        deploy_model(model)
        print("Pipeline executed successfully.")

    except Exception as e:
        print(f"Error in pipeline: {e}")
        return

if __name__ == "__main__":
    main_pipeline()