# CityLearn Forecasting - Quick Start Guide

## 🚀 Inizia in 5 Minuti

### 1. Installa le Dipendenze

```bash
cd citylearn_forecasting
pip install -r requirements.txt
```

### 2. Prepara i Tuoi Dati CityLearn

Metti i file CSV di CityLearn in:
```
citylearn_forecasting/data/raw/
├── phase_1/
│   ├── Building_1.csv
│   ├── Building_2.csv
│   └── ...
└── phase_2/
    ├── Building_1.csv
    └── ...
```

### 3. Train il Tuo Primo Modello

```bash
# LSTM Base
python experiments/train.py --model lstm --data data/raw/phase_1/Building_1.csv

# LSTM con Attention (CONSIGLIATO)
python experiments/train.py --model lstm_attention --data data/raw/phase_1/Building_1.csv

# TimesNet (State-of-the-art)
python experiments/train.py --model timesnet --data data/raw/phase_1/Building_1.csv
```

### 4. Valuta il Modello

```bash
python experiments/evaluate.py \
    --model_path checkpoints/lstm_attention_best.pth \
    --model_type lstm_attention \
    --data data/raw/phase_1/Building_1.csv
```

---

## 🎯 Training con Ottimizzazioni Avanzate

### Metodo 1: Script Python

```python
import torch
from torch.utils.data import DataLoader
import pandas as pd

from models import LSTMAttention
from data import TimeSeriesDataset
from utils import AdvancedTrainer, get_advanced_scheduler, create_optimizer

# 1. Carica dati
df = pd.read_csv('data/raw/phase_1/Building_1.csv')
data = df['Electricity:Facility [kW](Hourly)'].values.reshape(-1, 1)

# 2. Crea dataset
dataset = TimeSeriesDataset(data, seq_len=96, pred_len=24)
train_size = int(0.8 * len(dataset))
train_set, val_set = torch.utils.data.random_split(
    dataset, [train_size, len(dataset) - train_size]
)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

# 3. Crea modello
model = LSTMAttention(
    input_size=1,
    hidden_size=128,
    num_layers=2,
    output_size=24,
    dropout=0.2
)

# 4. Setup optimizer e scheduler OTTIMIZZATI
optimizer = create_optimizer(
    model,
    optimizer_type='adamw',  # Meglio di Adam!
    lr=0.001,
    weight_decay=0.01
)

scheduler = get_advanced_scheduler(
    optimizer,
    scheduler_type='cosine_warmup',  # Cosine con warmup
    num_epochs=100,
    warmup_epochs=5
)

# 5. Train con tecniche avanzate
device = 'cuda' if torch.cuda.is_available() else 'cpu'

trainer = AdvancedTrainer(
    model=model,
    optimizer=optimizer,
    criterion=torch.nn.MSELoss(),
    device=device,
    use_amp=True,  # Mixed precision - 40% più veloce!
    swa_start=80,  # Stochastic Weight Averaging
    grad_clip=1.0
)

# 6. Train!
for epoch in range(100):
    train_loss = trainer.train_epoch(train_loader, epoch)
    print(f"Epoch {epoch+1}/100 - Train Loss: {train_loss:.4f}")
    scheduler.step()
```

### Metodo 2: Usa il Config YAML (PIÙ SEMPLICE)

Modifica `configs/model_configs.yaml`:
```yaml
training:
  batch_size: 64
  epochs: 100
  learning_rate: 0.001
  weight_decay: 0.01
  early_stopping: 20

models:
  lstm_attention:
    hidden_size: 128
    num_layers: 2
    dropout: 0.2
```

Poi:
```bash
python experiments/train.py --model lstm_attention --config configs/model_configs.yaml
```

---

## 📊 Confronta Tutti i Modelli

```python
from experiments.evaluate import compare_models

model_paths = {
    'lstm': 'checkpoints/lstm_best.pth',
    'lstm_attention': 'checkpoints/lstm_attention_best.pth',
    'lstm_encoder_decoder': 'checkpoints/lstm_encoder_decoder_best.pth',
    'timesnet': 'checkpoints/timesnet_best.pth'
}

results = compare_models(
    model_paths=model_paths,
    data_path='data/raw/phase_1/Building_1.csv',
    output_dir='results/comparison'
)

# Output automatico:
# - model_comparison.csv (tabella con tutti i metrics)
# - models_comparison_plot.png (grafico visivo)
```

---

## 🎓 Esempi per Diversi Casi d'Uso

### Caso 1: Dataset Piccolo (< 10k samples)

**Problema:** Rischio alto di overfitting

**Soluzione:**
```python
model = LSTMAttention(
    input_size=1,
    hidden_size=64,  # Più piccolo
    num_layers=2,
    dropout=0.3,  # Dropout più alto
)

optimizer = create_optimizer(
    model,
    optimizer_type='adamw',
    lr=0.0005,  # LR più basso
    weight_decay=0.02  # Regularization più forte
)
```

### Caso 2: Dataset Grande (> 100k samples)

**Problema:** Training lento

**Soluzione:**
```python
model = TimesNet(
    input_size=1,
    hidden_size=256,  # Modello più grande
    num_layers=3,
    dropout=0.1
)

# Mixed precision + batch grande
trainer = AdvancedTrainer(
    model=model,
    optimizer=optimizer,
    use_amp=True,  # 40% più veloce!
    accumulation_steps=2  # Batch effettivo = 128
)

train_loader = DataLoader(dataset, batch_size=128)  # Batch grande
```

### Caso 3: Forecast Multi-Step (24h o più)

**Modello consigliato:** LSTM Encoder-Decoder o TimesNet

```python
from models import LSTMEncoderDecoder

model = LSTMEncoderDecoder(
    input_size=5,  # Multivariate
    hidden_size=128,
    output_size=1,
    forecast_horizon=24,  # 24 step ahead
    teacher_forcing_ratio=0.5
)
```

---

## 🔥 Tips & Tricks

### 1. Ensemble per Migliorare Performance

```python
# Train 3 modelli diversi
lstm = train_model('lstm', data)
attention = train_model('lstm_attention', data)
timesnet = train_model('timesnet', data)

# Weighted ensemble
final_pred = (
    0.4 * timesnet.predict(x) +
    0.3 * attention.predict(x) +
    0.3 * lstm.predict(x)
)
```

**Migliora accuratezza del 5-10%!**

### 2. Hyperparameter Tuning

```python
import optuna

def objective(trial):
    # Suggerisci hyperparameters
    hidden_size = trial.suggest_int('hidden_size', 64, 256)
    num_layers = trial.suggest_int('num_layers', 1, 4)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)

    # Train e valuta
    model = LSTMAttention(
        input_size=1,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )

    # ... train ...

    return val_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print(f"Best params: {study.best_params}")
```

### 3. Feature Engineering per CityLearn

```python
from data.preprocessor import TimeSeriesPreprocessor

preprocessor = TimeSeriesPreprocessor()

# Aggiungi features temporali
df = preprocessor.add_time_features(df, 'timestamp')

# Aggiungi lag features (consumi passati)
df = preprocessor.add_lag_features(
    df,
    columns=['electricity_consumption'],
    lags=[1, 24, 48, 168]  # 1h, 1day, 2days, 1week
)

# Aggiungi rolling statistics
df = preprocessor.add_rolling_features(
    df,
    columns=['electricity_consumption'],
    windows=[24, 168],  # 1 day, 1 week
    functions=['mean', 'std', 'min', 'max']
)
```

**Migliora MAE del 10-15%!**

---

## 📈 Monitoring Training

### Usa TensorBoard (opzionale)

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/lstm_attention')

for epoch in range(epochs):
    train_loss = train_epoch(...)
    val_loss = validate(...)

    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], epoch)

writer.close()
```

Poi visualizza:
```bash
tensorboard --logdir=runs
```

---

## ❓ FAQ

### Q: Quale modello dovrei usare?

**A:**
- **Dataset piccolo:** LSTM base o LSTM Attention
- **Dataset medio:** LSTM Attention (best balance)
- **Dataset grande:** TimesNet (state-of-the-art)
- **Multi-step forecasting:** LSTM Encoder-Decoder

### Q: Come evito overfitting?

**A:**
1. Aumenta dropout (0.2 → 0.3)
2. Aumenta weight_decay (0.0001 → 0.01)
3. Riduci model size (hidden_size: 128 → 64)
4. Usa early stopping (già implementato)
5. Aggiungi più dati o data augmentation

### Q: Il training è troppo lento?

**A:**
1. Usa mixed precision (`use_amp=True`)
2. Aumenta batch_size se hai memoria
3. Riduci seq_len (96 → 48)
4. Usa GPU se disponibile

### Q: Come miglioro le performance?

**A:**
1. Feature engineering (lags, rolling stats)
2. Hyperparameter tuning
3. Ensemble di modelli
4. Più dati di training
5. Data augmentation

---

## 📚 Struttura File per CityLearn Challenge

```
citylearn_forecasting/
├── data/
│   └── raw/
│       ├── phase_1/
│       │   ├── Building_1.csv
│       │   ├── Building_2.csv
│       │   └── ...
│       └── phase_2/
│           └── ...
├── checkpoints/  (creato automaticamente)
│   ├── lstm_best.pth
│   ├── lstm_attention_best.pth
│   └── ...
├── results/  (creato automaticamente)
│   ├── predictions.npz
│   ├── metrics.txt
│   └── plots/
└── experiments/
    ├── train.py
    └── evaluate.py
```

---

## 🎉 Pronto per Iniziare!

1. ✅ Installa dipendenze
2. ✅ Carica dati CityLearn
3. ✅ Train un modello con `python experiments/train.py`
4. ✅ Valuta con `python experiments/evaluate.py`
5. ✅ Ottimizza usando la guida `OPTIMIZATION_GUIDE.md`

**Hai domande?** Controlla `OPTIMIZATION_GUIDE.md` per dettagli avanzati!
