from xgboost import XGBClassifier

def get_model_config():
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    return model

if __name__ == "__main__":
    model = get_model_config()
    print("Model configuration:", model)