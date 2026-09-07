import mlflow
from sklearn.datasets import load_breast_cancer


def load_and_predict():
    MODEL_NAME = "cancer-classifier-prod"
    MODEL_ALIAS = "staging"

    cancer_data = load_breast_cancer(as_frame=True)
    target_names = cancer_data.target_names  # ['malignant', 'benign']

    try:
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    df = cancer_data.frame
    # ดึงตัวอย่างแรกของแต่ละคลาส (คลาส 0 และ คลาส 1)
    sample_0 = df[df["target"] == 0].iloc[0:1]
    sample_1 = df[df["target"] == 1].iloc[0:1]
    samples = pd.concat([sample_0, sample_1])

    X_sample = samples.drop("target", axis=1)
    y_actual = samples["target"].values

    predictions = model.predict(X_sample)

    print("-" * 40)
    for i in range(len(samples)):
        actual_name = target_names[y_actual[i]]
        pred_name = target_names[predictions[i]]
        is_correct = "Correct" if actual_name == pred_name else "Incorrect"
        print(f"Sample {i + 1}: Actual = {actual_name} | Predicted = {pred_name} [{is_correct}]")
    print("-" * 40)


if __name__ == "__main__":
    import pandas as pd

    load_and_predict()
