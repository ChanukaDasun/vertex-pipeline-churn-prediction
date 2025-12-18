import os
from google.cloud import storage
from google.cloud import aiplatform
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")
BUCKET_NAME = os.getenv("BUCKET_NAME")

def init_bucket_and_upload():

    # Initialize Vertex AI
    aiplatform.init(project=PROJECT_ID, location=REGION)
    
    # Create storage client
    storage_client = storage.Client(project=PROJECT_ID)
    
    try:
        # Get existing bucket (don't create if it exists)
        bucket = storage_client.get_bucket(BUCKET_NAME)
        print(f"✓ Connected to existing bucket: {BUCKET_NAME}")
    except Exception as e:
        print(f"✗ Error accessing bucket: {e}")
        return
    
    # Upload dataset
    try:
        local_file = "data/raw/customer_churn_dataset.csv"
        gcs_blob_path = "data/raw/customer_churn_dataset.csv"
        
        blob = bucket.blob(gcs_blob_path)
        blob.upload_from_filename(local_file)
        print(f"✓ Uploaded {local_file} to gs://{BUCKET_NAME}/{gcs_blob_path}")
        
    except FileNotFoundError:
        print(f"✗ File not found: {local_file}")
        print(f"   Current directory: {os.getcwd()}")
    except Exception as e:
        print(f"✗ Error uploading file: {e}")

    return storage_client, bucket

if __name__ == "__main__":
    init_bucket_and_upload()