# CityLearn Forecasting Project

## Overview
This project implements advanced time series forecasting models for CityLearn building energy management. The goal is to predict building energy consumption, solar generation, and other key variables to optimize energy efficiency.

## Models Implemented

### 1. LSTM (Long Short-Term Memory)
Basic LSTM architecture for univariate and multivariate time series forecasting.

### 2. LSTM Autoencoder
Encoder-decoder architecture that learns compressed representations of time series data for anomaly detection and forecasting.

### 3. LSTM with Attention
LSTM enhanced with attention mechanism to focus on relevant time steps for better prediction accuracy.

### 4. LSTM Encoder-Decoder
Sequence-to-sequence architecture for multi-step ahead forecasting.

### 5. TimesNet
State-of-the-art temporal 2D-variation modeling for time series forecasting, based on the TimesNet paper.

## Project Structure

```
citylearn_forecasting/
├── models/              # Model implementations
│   ├── lstm.py
│   ├── lstm_autoencoder.py
│   ├── lstm_attention.py
│   ├── lstm_encoder_decoder.py
│   └── timesnet.py
├── data/               # Data loading and preprocessing
│   ├── dataset.py
│   └── preprocessor.py
├── utils/              # Utility functions
│   ├── metrics.py
│   ├── visualization.py
│   └── trainer.py
├── configs/            # Configuration files
│   └── model_configs.yaml
├── experiments/        # Experiment scripts
│   ├── train.py
│   └── evaluate.py
├── notebooks/          # Jupyter notebooks for analysis
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training a Model

```python
from experiments.train import train_model

# Train LSTM model
train_model(model_type='lstm', config_path='configs/model_configs.yaml')

# Train LSTM Attention model
train_model(model_type='lstm_attention', config_path='configs/model_configs.yaml')
```

### Evaluation

```python
from experiments.evaluate import evaluate_model

results = evaluate_model(model_path='path/to/model.pth', test_data='path/to/test.csv')
```

## Features

- Multiple state-of-the-art forecasting models
- Comprehensive data preprocessing pipeline
- Training and evaluation framework
- Visualization tools
- Model comparison utilities
- Support for CityLearn environment

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- CityLearn

## References

- TimesNet: [Temporal 2D-Variation Modeling for General Time Series Analysis](https://arxiv.org/abs/2210.02186)
- CityLearn: [CityLearn: Diverse Gym Environments for Reinforcement Learning in Cities](https://github.com/intelligent-environments-lab/CityLearn)

## License

MIT License
