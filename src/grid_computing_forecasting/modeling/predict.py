import os
import joblib
import pandas as pd


def with_xgboost(path_model: str, model_name: str, test_row: pd.Series):
    """
    Loads a trained model from a .pkl file and predicts the target value for a single row of data.

    Parameters:
    model_path (str): The file path to the saved model (.pkl).
    test_row (pd.DataFrame or pd.Series): A single row of data for prediction.

    Returns:
    float: The predicted value.
    """
    # Ensure the path exists
    os.makedirs(path_model, exist_ok=True)

    # Construct the full file path
    model_file = os.path.join(path_model, model_name)
    # Load the model from the specified path
    model = joblib.load(model_file)

    # Ensure the test_row is a 2D array or DataFrame (required by the model)
    if isinstance(test_row, pd.Series):
        test_row = test_row.values.reshape(1, -1)
    elif isinstance(test_row, pd.DataFrame):
        test_row = test_row.to_numpy()

    # Make the prediction
    prediction = model.predict(test_row)

    # Return the predicted value (assuming it's a regression problem)
    return prediction[0]
