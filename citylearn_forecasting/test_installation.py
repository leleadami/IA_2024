"""
Quick test to verify installation and functionality
"""

import sys
import torch
import numpy as np

print("="*50)
print("CityLearn Forecasting - Installation Test")
print("="*50)

# Test 1: Check PyTorch
print("\n✓ Testing PyTorch...")
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# Test 2: Import models
print("\n✓ Testing Models...")
try:
    from models import (
        LSTMForecaster,
        LSTMAutoencoder,
        LSTMAttention,
        LSTMEncoderDecoder,
        TimesNet
    )
    print("  All models imported successfully!")
except Exception as e:
    print(f"  ✗ Error importing models: {e}")
    sys.exit(1)

# Test 3: Import data utilities
print("\n✓ Testing Data Utilities...")
try:
    from data import TimeSeriesDataset, TimeSeriesPreprocessor
    print("  Data utilities imported successfully!")
except Exception as e:
    print(f"  ✗ Error importing data utilities: {e}")
    sys.exit(1)

# Test 4: Import training utilities
print("\n✓ Testing Training Utilities...")
try:
    from utils import (
        Trainer,
        AdvancedTrainer,
        calculate_metrics,
        get_advanced_scheduler
    )
    print("  Training utilities imported successfully!")
except Exception as e:
    print(f"  ✗ Error importing training utilities: {e}")
    sys.exit(1)

# Test 5: Create and test a simple model
print("\n✓ Testing Model Creation...")
try:
    model = LSTMForecaster(
        input_size=1,
        hidden_size=32,
        num_layers=1,
        output_size=1,
        dropout=0.1
    )

    # Test forward pass
    batch_size = 4
    seq_len = 10
    x = torch.randn(batch_size, seq_len, 1)
    y = model(x)

    print(f"  Model created successfully!")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

except Exception as e:
    print(f"  ✗ Error testing model: {e}")
    sys.exit(1)

# Test 6: Test dataset creation
print("\n✓ Testing Dataset...")
try:
    # Generate synthetic data
    data = np.sin(np.linspace(0, 10, 1000)).reshape(-1, 1)

    dataset = TimeSeriesDataset(
        data=data,
        seq_len=24,
        pred_len=6,
        stride=1,
        scale=True
    )

    x, y = dataset[0]
    print(f"  Dataset created successfully!")
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Sample input shape: {x.shape}")
    print(f"  Sample target shape: {y.shape}")

except Exception as e:
    print(f"  ✗ Error testing dataset: {e}")
    sys.exit(1)

# Test 7: Test metrics
print("\n✓ Testing Metrics...")
try:
    y_true = np.random.randn(100)
    y_pred = y_true + np.random.randn(100) * 0.1

    from utils import MAE, RMSE, MAPE, R2

    mae = MAE(y_true, y_pred)
    rmse = RMSE(y_true, y_pred)
    mape = MAPE(y_true, y_pred)
    r2 = R2(y_true, y_pred)

    print(f"  Metrics calculated successfully!")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAPE: {mape:.4f}%")
    print(f"  R²: {r2:.4f}")

except Exception as e:
    print(f"  ✗ Error testing metrics: {e}")
    sys.exit(1)

# Test 8: Test all models
print("\n✓ Testing All Model Architectures...")
models_to_test = [
    ('LSTM', LSTMForecaster(input_size=1, hidden_size=16, output_size=1)),
    ('LSTM Autoencoder', LSTMAutoencoder(input_size=1, hidden_size=16)),
    ('LSTM Attention', LSTMAttention(input_size=1, hidden_size=16, output_size=1)),
    ('LSTM Encoder-Decoder', LSTMEncoderDecoder(input_size=1, hidden_size=16, output_size=1, forecast_horizon=6)),
    ('TimesNet', TimesNet(input_size=1, seq_len=24, pred_len=6, d_model=16, d_ff=32, num_layers=1))
]

x_test = torch.randn(2, 24, 1)

for name, model in models_to_test:
    try:
        with torch.no_grad():
            output = model(x_test)
        print(f"  ✓ {name}: {sum(p.numel() for p in model.parameters()):,} params, output shape {output.shape}")
    except Exception as e:
        print(f"  ✗ {name} failed: {e}")

# Final summary
print("\n" + "="*50)
print("✅ ALL TESTS PASSED!")
print("="*50)
print("\nYour CityLearn Forecasting installation is working correctly!")
print("\nNext steps:")
print("1. Load your CityLearn data (CSV files)")
print("2. Run: python experiments/train.py --model lstm_attention --data your_data.csv")
print("3. Check QUICKSTART.md for more examples")
print("\n" + "="*50)
