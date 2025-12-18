from bucket_init import BUCKET_NAME
from google.cloud import aiplatform

def automl_train():
    """Train an AutoML model using the dataset in GCS."""
    # Initialize Vertex AI
    aiplatform.init(project="sodium-binder-426309-f8", location="us-central1")

    try:
        # Create a tabular dataset
        dataset = aiplatform.TabularDataset.create(
            display_name="churn-dataset",
            gcs_source=f"gs://{BUCKET_NAME}/data/raw/customer_churn_dataset.csv",
        )

        print(f"Dataset created: {dataset.resource_name}")
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return

    try:
        # Define training job
        job = aiplatform.AutoMLTabularTrainingJob(
            display_name="churn-prediction-automl",
            optimization_prediction_type="classification",
            optimization_objective="maximize-au-prc",  # or "maximize-au-roc"
        )

        # Train the model
        model = job.run(
            dataset=dataset,
            target_column="churn",  # Your label column name
            training_fraction_split=0.8,
            validation_fraction_split=0.1,
            test_fraction_split=0.1,
            budget_milli_node_hours=1000,  # 1 hour (1000 milli-node-hours)
            model_display_name="churn-prediction-model",
        )

        print(f"Model trained: {model.resource_name}")
    except Exception as e:
        print(f"Error during training: {e}")
        return

if __name__ == "__main__":
    automl_train()