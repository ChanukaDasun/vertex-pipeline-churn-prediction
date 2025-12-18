from google.cloud import aiplatform
import os
import joblib
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")
BUCKET_NAME = os.getenv("BUCKET_NAME")

def deploy_model(sklearn_model):
    """Save, upload, and deploy the trained model to Vertex AI."""
    aiplatform.init(project=PROJECT_ID, location=REGION)
    
    try:
        # Step 1: Save the model locally
        local_model_path = "models/model.pkl"
        os.makedirs("models", exist_ok=True)
        joblib.dump(sklearn_model, local_model_path)
        print(f"Model saved locally to {local_model_path}")
        
        # Step 2: Upload model to GCS
        from google.cloud import storage
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(BUCKET_NAME)
        
        gcs_model_path = "models/model.pkl"
        blob = bucket.blob(gcs_model_path)
        blob.upload_from_filename(local_model_path)
        print(f"Model uploaded to gs://{BUCKET_NAME}/{gcs_model_path}")
        
        # Step 3: Upload to Vertex AI Model Registry
        model = aiplatform.Model.upload(
            display_name="churn-prediction-model",
            artifact_uri=f"gs://{BUCKET_NAME}/models/",
            serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
        )
        print(f"Model uploaded to Vertex AI: {model.resource_name}")
        
        # Step 4: Create an endpoint
        endpoint = aiplatform.Endpoint.create(
            display_name="churn-prediction-endpoint",
        )
        print(f"Endpoint created: {endpoint.resource_name}")
        
        # Step 5: Deploy the model
        model.deploy(
            endpoint=endpoint,
            deployed_model_display_name="churn-model-v1",
            machine_type="n1-standard-2",
            min_replica_count=1,
            max_replica_count=3,
        )
        
        print(f"Model deployed successfully to: {endpoint.resource_name}")
        return endpoint
        
    except Exception as e:
        print(f"✗ Error deploying model: {e}")
        raise

if __name__ == "__main__":
    # This won't work standalone anymore - needs a trained model
    print("This script should be called from the pipeline with a trained model")