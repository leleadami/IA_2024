"""
Data preprocessing utilities for time series
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Optional, Tuple, List


class TimeSeriesPreprocessor:
    """
    Comprehensive preprocessor for time series data

    Features:
    - Multiple scaling methods
    - Missing value handling
    - Outlier detection and treatment
    - Feature engineering (time features, lags, rolling statistics)
    - Detrending and deseasonalization

    Args:
        scaler_type (str): Type of scaler ('standard', 'minmax', 'robust')
        handle_missing (str): Method to handle missing values ('drop', 'forward_fill', 'interpolate')
        handle_outliers (str): Method to handle outliers ('clip', 'remove', 'none')
        outlier_threshold (float): Number of standard deviations for outlier detection
    """

    def __init__(
        self,
        scaler_type: str = 'standard',
        handle_missing: str = 'forward_fill',
        handle_outliers: str = 'clip',
        outlier_threshold: float = 3.0
    ):
        self.scaler_type = scaler_type
        self.handle_missing = handle_missing
        self.handle_outliers = handle_outliers
        self.outlier_threshold = outlier_threshold

        # Initialize scaler
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")

        self.fitted = False

    def fit(self, data):
        """
        Fit the preprocessor on training data

        Args:
            data (np.ndarray or pd.DataFrame): Training data
        """
        if isinstance(data, pd.DataFrame):
            data = data.values

        # Handle missing values
        data = self._handle_missing_values(data)

        # Handle outliers
        data = self._handle_outliers(data)

        # Fit scaler
        self.scaler.fit(data)
        self.fitted = True

        return self

    def transform(self, data):
        """
        Transform data using fitted preprocessor

        Args:
            data (np.ndarray or pd.DataFrame): Data to transform

        Returns:
            Transformed data
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")

        is_dataframe = isinstance(data, pd.DataFrame)
        if is_dataframe:
            columns = data.columns
            index = data.index
            data = data.values

        # Handle missing values
        data = self._handle_missing_values(data)

        # Handle outliers
        data = self._handle_outliers(data)

        # Scale
        data = self.scaler.transform(data)

        if is_dataframe:
            data = pd.DataFrame(data, columns=columns, index=index)

        return data

    def fit_transform(self, data):
        """Fit and transform in one step"""
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        """
        Inverse transform scaled data back to original scale

        Args:
            data: Scaled data

        Returns:
            Original scale data
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before inverse_transform")

        is_dataframe = isinstance(data, pd.DataFrame)
        if is_dataframe:
            columns = data.columns
            index = data.index
            data = data.values

        data = self.scaler.inverse_transform(data)

        if is_dataframe:
            data = pd.DataFrame(data, columns=columns, index=index)

        return data

    def _handle_missing_values(self, data):
        """Handle missing values in data"""
        if not np.any(np.isnan(data)):
            return data

        if self.handle_missing == 'drop':
            # Remove rows with missing values
            mask = ~np.isnan(data).any(axis=1)
            return data[mask]

        elif self.handle_missing == 'forward_fill':
            # Forward fill
            if isinstance(data, pd.DataFrame):
                return data.fillna(method='ffill').fillna(method='bfill')
            else:
                df = pd.DataFrame(data)
                return df.fillna(method='ffill').fillna(method='bfill').values

        elif self.handle_missing == 'interpolate':
            # Linear interpolation
            if isinstance(data, pd.DataFrame):
                return data.interpolate(method='linear', limit_direction='both')
            else:
                df = pd.DataFrame(data)
                return df.interpolate(method='linear', limit_direction='both').values

        return data

    def _handle_outliers(self, data):
        """Handle outliers in data"""
        if self.handle_outliers == 'none':
            return data

        # Calculate z-scores
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        z_scores = np.abs((data - mean) / (std + 1e-8))

        if self.handle_outliers == 'clip':
            # Clip outliers to threshold
            lower_bound = mean - self.outlier_threshold * std
            upper_bound = mean + self.outlier_threshold * std
            return np.clip(data, lower_bound, upper_bound)

        elif self.handle_outliers == 'remove':
            # Remove rows with outliers
            mask = np.all(z_scores < self.outlier_threshold, axis=1)
            return data[mask]

        return data

    @staticmethod
    def add_time_features(df: pd.DataFrame, timestamp_column: str = 'timestamp') -> pd.DataFrame:
        """
        Add time-based features to dataframe

        Args:
            df: DataFrame with timestamp column
            timestamp_column: Name of timestamp column

        Returns:
            DataFrame with additional time features
        """
        df = df.copy()

        if timestamp_column not in df.columns:
            raise ValueError(f"Column {timestamp_column} not found in dataframe")

        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
            df[timestamp_column] = pd.to_datetime(df[timestamp_column])

        # Extract time features
        df['hour'] = df[timestamp_column].dt.hour
        df['day'] = df[timestamp_column].dt.day
        df['month'] = df[timestamp_column].dt.month
        df['day_of_week'] = df[timestamp_column].dt.dayofweek
        df['day_of_year'] = df[timestamp_column].dt.dayofyear
        df['week_of_year'] = df[timestamp_column].dt.isocalendar().week

        # Cyclical encoding for periodic features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Boolean features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        return df

    @staticmethod
    def add_lag_features(
        df: pd.DataFrame,
        columns: List[str],
        lags: List[int]
    ) -> pd.DataFrame:
        """
        Add lagged features

        Args:
            df: DataFrame
            columns: Columns to create lags for
            lags: List of lag values

        Returns:
            DataFrame with lag features
        """
        df = df.copy()

        for col in columns:
            if col not in df.columns:
                continue

            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)

        return df

    @staticmethod
    def add_rolling_features(
        df: pd.DataFrame,
        columns: List[str],
        windows: List[int],
        functions: List[str] = ['mean', 'std', 'min', 'max']
    ) -> pd.DataFrame:
        """
        Add rolling window statistics

        Args:
            df: DataFrame
            columns: Columns to compute rolling statistics for
            windows: List of window sizes
            functions: List of aggregation functions

        Returns:
            DataFrame with rolling features
        """
        df = df.copy()

        for col in columns:
            if col not in df.columns:
                continue

            for window in windows:
                for func in functions:
                    if func == 'mean':
                        df[f'{col}_rolling_{window}_mean'] = df[col].rolling(window).mean()
                    elif func == 'std':
                        df[f'{col}_rolling_{window}_std'] = df[col].rolling(window).std()
                    elif func == 'min':
                        df[f'{col}_rolling_{window}_min'] = df[col].rolling(window).min()
                    elif func == 'max':
                        df[f'{col}_rolling_{window}_max'] = df[col].rolling(window).max()

        return df

    @staticmethod
    def detrend(data: np.ndarray, method: str = 'linear') -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove trend from time series

        Args:
            data: Time series data
            method: Detrending method ('linear', 'polynomial')

        Returns:
            detrended_data, trend
        """
        if method == 'linear':
            t = np.arange(len(data))
            coeffs = np.polyfit(t, data, 1)
            trend = np.polyval(coeffs, t)
        else:
            raise ValueError(f"Unknown detrending method: {method}")

        detrended = data - trend
        return detrended, trend

    @staticmethod
    def difference(data: np.ndarray, order: int = 1) -> np.ndarray:
        """
        Apply differencing to make series stationary

        Args:
            data: Time series data
            order: Order of differencing

        Returns:
            Differenced data
        """
        result = data.copy()
        for _ in range(order):
            result = np.diff(result)
        return result

    @staticmethod
    def split_sequences(
        sequences: np.ndarray,
        n_steps_in: int,
        n_steps_out: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split sequences into input and output samples

        Args:
            sequences: Time series sequences
            n_steps_in: Number of input time steps
            n_steps_out: Number of output time steps

        Returns:
            X, y arrays
        """
        X, y = [], []

        for i in range(len(sequences)):
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out

            if out_end_ix > len(sequences):
                break

            X.append(sequences[i:end_ix])
            y.append(sequences[end_ix:out_end_ix])

        return np.array(X), np.array(y)
