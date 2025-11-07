"""
Dataset classes for time series forecasting
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, List


class TimeSeriesDataset(Dataset):
    """
    Generic time series dataset for forecasting tasks

    Args:
        data (np.ndarray or pd.DataFrame): Time series data
        seq_len (int): Length of input sequence
        pred_len (int): Length of prediction (forecast horizon)
        stride (int): Stride for sliding window
        features (list): List of feature column names (for DataFrame)
        target (str): Target column name (for DataFrame)
        scale (bool): Whether to normalize the data
    """

    def __init__(
        self,
        data,
        seq_len: int = 96,
        pred_len: int = 24,
        stride: int = 1,
        features: Optional[List[str]] = None,
        target: Optional[str] = None,
        scale: bool = True
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride

        # Convert to numpy array if pandas DataFrame
        if isinstance(data, pd.DataFrame):
            if features is not None:
                self.feature_columns = features
                data_features = data[features].values
            else:
                self.feature_columns = data.columns.tolist()
                data_features = data.values

            if target is not None:
                self.target_column = target
                data_target = data[target].values
            else:
                data_target = data_features[:, 0]

            self.data = data_features
            self.targets = data_target
        else:
            self.data = np.array(data)
            if len(self.data.shape) == 1:
                self.data = self.data.reshape(-1, 1)
            self.targets = self.data[:, 0]

        # Scaling
        self.scale = scale
        if scale:
            self.mean = np.mean(self.data, axis=0)
            self.std = np.std(self.data, axis=0) + 1e-8
            self.data = (self.data - self.mean) / self.std

            self.target_mean = np.mean(self.targets)
            self.target_std = np.std(self.targets) + 1e-8
            self.targets = (self.targets - self.target_mean) / self.target_std

        # Create sliding windows
        self.indices = self._create_indices()

    def _create_indices(self):
        """Create indices for sliding windows"""
        indices = []
        max_idx = len(self.data) - self.seq_len - self.pred_len + 1

        for i in range(0, max_idx, self.stride):
            indices.append(i)

        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        end_idx = start_idx + self.seq_len

        # Input sequence
        x = self.data[start_idx:end_idx]

        # Target sequence
        target_start = end_idx
        target_end = target_start + self.pred_len
        y = self.targets[target_start:target_end]

        return torch.FloatTensor(x), torch.FloatTensor(y)

    def inverse_transform(self, data, target=True):
        """
        Inverse transform normalized data back to original scale

        Args:
            data: Normalized data
            target: If True, use target statistics, else use feature statistics

        Returns:
            Original scale data
        """
        if not self.scale:
            return data

        if target:
            return data * self.target_std + self.target_mean
        else:
            return data * self.std + self.mean


class CityLearnDataset(Dataset):
    """
    Dataset specifically for CityLearn environment data

    Args:
        env_data (dict): Dictionary containing building data from CityLearn
        seq_len (int): Length of input sequence
        pred_len (int): Length of prediction
        building_ids (list): List of building IDs to include
        features (list): List of features to use
        target (str): Target variable to predict
    """

    def __init__(
        self,
        env_data: dict,
        seq_len: int = 96,
        pred_len: int = 24,
        stride: int = 1,
        building_ids: Optional[List[str]] = None,
        features: Optional[List[str]] = None,
        target: str = 'net_electricity_consumption'
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride
        self.target = target

        # Default features if not specified
        if features is None:
            features = [
                'net_electricity_consumption',
                'solar_generation',
                'cooling_demand',
                'heating_demand',
                'outdoor_dry_bulb_temperature',
                'hour',
                'day_type'
            ]

        self.features = features

        # Process building data
        if building_ids is None:
            building_ids = list(env_data.keys())

        self.building_ids = building_ids

        # Combine data from all buildings
        all_data = []
        all_targets = []

        for building_id in building_ids:
            building_data = env_data[building_id]

            # Extract features
            feature_data = []
            for feature in features:
                if feature in building_data:
                    feature_data.append(building_data[feature])
                else:
                    # If feature not available, use zeros
                    feature_data.append(np.zeros(len(building_data[list(building_data.keys())[0]])))

            feature_data = np.column_stack(feature_data)

            # Extract target
            if target in building_data:
                target_data = building_data[target]
            else:
                target_data = feature_data[:, 0]

            all_data.append(feature_data)
            all_targets.append(target_data)

        # Concatenate data from all buildings
        self.data = np.vstack(all_data)
        self.targets = np.concatenate(all_targets)

        # Normalize
        self.mean = np.mean(self.data, axis=0)
        self.std = np.std(self.data, axis=0) + 1e-8
        self.data = (self.data - self.mean) / self.std

        self.target_mean = np.mean(self.targets)
        self.target_std = np.std(self.targets) + 1e-8
        self.targets = (self.targets - self.target_mean) / self.target_std

        # Create indices
        self.indices = self._create_indices()

    def _create_indices(self):
        """Create indices for sliding windows"""
        indices = []
        max_idx = len(self.data) - self.seq_len - self.pred_len + 1

        for i in range(0, max_idx, self.stride):
            indices.append(i)

        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        end_idx = start_idx + self.seq_len

        # Input sequence
        x = self.data[start_idx:end_idx]

        # Target sequence
        target_start = end_idx
        target_end = target_start + self.pred_len
        y = self.targets[target_start:target_end]

        return torch.FloatTensor(x), torch.FloatTensor(y)

    def inverse_transform(self, data, target=True):
        """Inverse transform to original scale"""
        if target:
            return data * self.target_std + self.target_mean
        else:
            return data * self.std + self.mean


class MultiVariateTimeSeriesDataset(Dataset):
    """
    Dataset for multivariate time series forecasting
    where multiple variables are predicted simultaneously

    Args:
        data (np.ndarray): Time series data (time_steps, features)
        seq_len (int): Input sequence length
        pred_len (int): Prediction length
        stride (int): Stride for sliding window
        predict_all_features (bool): If True, predict all features; else only first feature
    """

    def __init__(
        self,
        data: np.ndarray,
        seq_len: int = 96,
        pred_len: int = 24,
        stride: int = 1,
        predict_all_features: bool = True
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride
        self.predict_all_features = predict_all_features

        # Ensure 2D array
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        self.data = data
        self.n_features = data.shape[1]

        # Normalize
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0) + 1e-8
        self.data = (data - self.mean) / self.std

        # Create indices
        self.indices = self._create_indices()

    def _create_indices(self):
        indices = []
        max_idx = len(self.data) - self.seq_len - self.pred_len + 1

        for i in range(0, max_idx, self.stride):
            indices.append(i)

        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        end_idx = start_idx + self.seq_len

        # Input sequence
        x = self.data[start_idx:end_idx]

        # Target sequence
        target_start = end_idx
        target_end = target_start + self.pred_len

        if self.predict_all_features:
            y = self.data[target_start:target_end]
        else:
            # Only predict first feature
            y = self.data[target_start:target_end, 0]

        return torch.FloatTensor(x), torch.FloatTensor(y)

    def inverse_transform(self, data):
        """Inverse transform to original scale"""
        return data * self.std + self.mean


def create_train_val_test_split(
    dataset: Dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split dataset into train, validation, and test sets

    Args:
        dataset: PyTorch Dataset
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    total_len = len(dataset)
    train_len = int(total_len * train_ratio)
    val_len = int(total_len * val_ratio)
    test_len = total_len - train_len - val_len

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_len, val_len, test_len]
    )

    return train_dataset, val_dataset, test_dataset
