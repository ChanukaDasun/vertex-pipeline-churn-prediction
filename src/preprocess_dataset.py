import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(dataset_path):
    """Preprocess the dataset for training."""

    print("Preprocessing dataset...")

    try:
        print("dataset:", dataset_path)
        df = pd.read_csv(dataset_path)
        print("Initial data shape:", df.info())

        # one hot encode 3 object rows
        encoded_df = pd.get_dummies(
            df,
            columns=["Gender", "Subscription Type", "Contract Length"],
            drop_first=False
        )
        encoded_df.info()

        from sklearn.model_selection import train_test_split

        X = encoded_df.drop(["Churn", "CustomerID"], axis=1)
        y = encoded_df["Churn"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None
