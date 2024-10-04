import os
import joblib
import torch
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_absolute_error
from grid_computing_forecasting import config


def xgboost(df: pd.DataFrame, path_save: str, model_name: str):
    """
    Trains an XGBoost model on a given dataset, tunes hyperparameters, and saves the best model.

    Parameters:
    df (pd.DataFrame): The dataset including features and the target variable 'RunTime'.
    path_save (str): The path to save the trained model.
    model_name (str): The name for the saved model file (without extension).

    Returns:
    best_model (XGBRegressor): The best XGBoost model after hyperparameter tuning.
    """
    # Define the size of the training set (57% of the data)
    train_size = int(len(df) * 0.57)

    # Split the data into training and testing sets based on the timestamp
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]

    # Split train data into input (X_train) and target (y_train) columns
    X_train = train_data.drop(columns=['Timestamp', 'RunTime'])
    y_train = train_data['RunTime']

    # Split test data into input (X_test) and target (y_test) columns
    X_test = test_data.drop(columns=['Timestamp', 'RunTime'])
    y_test = test_data['RunTime']

    # Define the hyperparameters grid to search
    param_grid = {
        'learning_rate': [0.01],  # Learning rate variations
        'n_estimators': [140],    # Number of trees in the forest
        'max_depth': [4],         # Maximum depth of each tree
        'gamma': [0.01],          # Minimum loss reduction required to make a further partition
    }

    # Define the base model
    xgboost_model = XGBRegressor()

    # Set up TimeSeriesSplit to respect the temporal order
    tscv = TimeSeriesSplit(n_splits=5)

    # Set up GridSearchCV with the model and hyperparameter grid
    grid_search = GridSearchCV(estimator=xgboost_model, param_grid=param_grid,
                               scoring='neg_mean_absolute_error', cv=tscv, n_jobs=-1, verbose=1)

    # Fit the model
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters found by the search
    print("Best hyperparameters:", grid_search.best_params_)

    # Print the best cross-validation MAE score
    print("Best cross-validation MAE:", -grid_search.best_score_)

    # Get the best model from the search
    best_model = grid_search.best_estimator_

    # Make predictions on the test data
    y_pred = best_model.predict(X_test)

    # Evaluate the performance on the test set
    MAE_test = mean_absolute_error(y_test, y_pred)

    # Print the test MAE
    print("Test MAE score with optimized hyperparameters:", MAE_test)

    # Ensure the path exists
    os.makedirs(path_save, exist_ok=True)

    # Construct the full file path
    model_file = os.path.join(path_save, f"{model_name}.pkl")

    # Save the best model
    joblib.dump(best_model, model_file)
    print(f"Model saved to {model_file}")

    # Return the best model
    return best_model


# Grownet model
class MoADataset:
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return (self.features.shape[0])

    def __getitem__(self, idx):
        dct = {
            'x': torch.tensor(self.features[idx, :], dtype=torch.float),
            'y': torch.tensor(self.targets[idx, :], dtype=torch.float)
        }
        return dct


class TestDataset:
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return (self.features.shape[0])

    def __getitem__(self, idx):
        dct = {
            'x': torch.tensor(self.features[idx, :], dtype=torch.float)
        }
        return dct


def grownet(df: pd.DataFrame):
    df = df.drop(columns=['Timestamp'])
    # Split the DataFrame into train and test sets
    df_train, df_test = train_test_split(df, test_size=0.43, random_state=42)

    # Convert DataFrame to NPZ format
    np.savez('./../data/interim/dataset_tr.npz',
             features=(df_train.drop(columns=['RunTime'])).to_numpy(),
             labels=(df_train[['RunTime']]).to_numpy())
    np.savez('./../data/interim/dataset_te.npz',
             features=(df_test.drop(columns=['RunTime'])).to_numpy(),
             labels=(df_test[['RunTime']]).to_numpy())

    return None
