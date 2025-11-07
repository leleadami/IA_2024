# 🎓 Guida Completa per la Tesi

## 📋 Checklist Completa

### ✅ Parte 1: Setup e Preparazione

- [ ] Installa le dipendenze: `pip install -r requirements.txt`
- [ ] Testa l'installazione: `python test_installation.py`
- [ ] Scarica dati CityLearn Phase 1 e 2 (2023)
- [ ] Metti i dati in `data/raw/phase_1/` e `data/raw/phase_2/`

### ✅ Parte 2: Esperimenti

#### Step 1: Run Tutti gli Esperimenti (1 comando!)

```bash
python experiments/run_all_experiments.py --data data/raw/phase_1/Building_1.csv --output thesis_results
```

Questo script:
- ✓ Train tutti i 5 modelli automaticamente
- ✓ Evalua su test set
- ✓ Salva tutte le metriche
- ✓ Crea tabelle comparative
- ✓ Genera report completo

**Tempo stimato:** 2-4 ore (dipende dalla GPU)

#### Step 2: Genera Tutte le Figure per la Tesi

```bash
python experiments/generate_thesis_figures.py --results_dir thesis_results --output_dir thesis_results/figures
```

Genera automaticamente:
1. ✓ Confronto metriche (bar chart)
2. ✓ Accuracy vs Complexity (scatter plot)
3. ✓ Training time comparison
4. ✓ Radar chart multi-metric
5. ✓ Pareto frontier
6. ✓ Summary table

**Output:** Figure pronte per LaTeX/Word (PNG 300 DPI)

### ✅ Parte 3: Analisi Risultati

Controlla i file generati in `thesis_results/`:
- [ ] `all_results.json` - Tutti i risultati in formato JSON
- [ ] `comparison_table.csv` - Tabella comparativa (Excel/CSV)
- [ ] `comparison_table.tex` - Tabella per LaTeX
- [ ] `summary_report.txt` - Report testuale completo
- [ ] `figures/` - Tutte le figure per la tesi

### ✅ Parte 4: Scrittura Tesi

1. **Leggi la struttura:** Apri `THESIS_STRUCTURE.md`
   - Contiene la struttura completa capitolo per capitolo
   - Include lunghezza pagine consigliate
   - Suggerimenti per ogni sezione

2. **Inizia a scrivere:**
   - Capitolo 2 (Background) - puoi iniziare subito
   - Capitolo 3 (Methodology) - descrivi il codice
   - Capitolo 4 (Results) - usa i risultati degli esperimenti
   - Capitolo 1 (Introduction) - scrivi per ultimo
   - Capitolo 6 (Conclusions) - scrivi per ultimo

3. **Usa i materiali generati:**
   - Tabelle da `comparison_table.tex` o `.csv`
   - Figure da `figures/`
   - Metriche da `summary_report.txt`

---

## 🚀 Quick Start per Esperimenti

### Opzione A: Tutto Automatico (CONSIGLIATO)

```bash
# 1. Run tutti gli esperimenti
python experiments/run_all_experiments.py \
    --data data/raw/phase_1/Building_1.csv \
    --output thesis_results

# 2. Genera figure
python experiments/generate_thesis_figures.py \
    --results_dir thesis_results

# 3. FATTO! Guarda i risultati in thesis_results/
```

### Opzione B: Manuale (per controllo fine)

```bash
# Train singoli modelli
python experiments/train.py --model lstm --data your_data.csv
python experiments/train.py --model lstm_attention --data your_data.csv
python experiments/train.py --model timesnet --data your_data.csv

# Evaluate
python experiments/evaluate.py \
    --model_path checkpoints/lstm_best.pth \
    --model_type lstm \
    --data your_data.csv
```

---

## 📊 Cosa Include il Progetto per la Tesi

### 1. Implementazioni Complete
- ✅ **5 modelli state-of-the-art**
  - LSTM Base
  - LSTM Autoencoder
  - LSTM Attention (Bahdanau + Luong + Self-Attention)
  - LSTM Encoder-Decoder
  - TimesNet

- ✅ **Best practices già implementate**
  - Dropout ottimizzato
  - Early stopping
  - Learning rate scheduling
  - Gradient clipping
  - Weight decay
  - Mixed precision training
  - Advanced optimizers (AdamW)

### 2. Framework Completo
- ✅ Data preprocessing robusto
- ✅ Training e evaluation pipeline
- ✅ Comprehensive metrics (MAE, RMSE, MAPE, R², ecc.)
- ✅ Visualization tools
- ✅ Test suite

### 3. Documentazione
- ✅ `README.md` - Overview generale
- ✅ `QUICKSTART.md` - Guida rapida esempi
- ✅ `OPTIMIZATION_GUIDE.md` - Guida ottimizzazioni
- ✅ `THESIS_STRUCTURE.md` - Struttura tesi completa
- ✅ `THESIS_README.md` (questo file) - Guida tesi

### 4. Script Automatici
- ✅ `run_all_experiments.py` - Run esperimenti completi
- ✅ `generate_thesis_figures.py` - Genera figure automaticamente
- ✅ `test_installation.py` - Verifica installazione

---

## 📈 Output degli Esperimenti

Dopo aver eseguito `run_all_experiments.py`, otterrai:

```
thesis_results/
├── all_results.json              # Tutti i risultati
├── comparison_table.csv          # Tabella per Excel
├── comparison_table.tex          # Tabella per LaTeX
├── summary_report.txt            # Report testuale completo
├── checkpoints/                  # Modelli trainati
│   ├── lstm_best.pth
│   ├── lstm_attention_best.pth
│   ├── lstm_encoder_decoder_best.pth
│   ├── lstm_autoencoder_best.pth
│   └── timesnet_best.pth
└── [model_name]/                 # Per ogni modello:
    ├── predictions.npz           # Predizioni salvate
    ├── metrics.txt               # Metriche dettagliate
    ├── predictions_plot.png      # Grafico predizioni
    └── residuals.png             # Analisi residui
```

Dopo `generate_thesis_figures.py`:

```
thesis_results/figures/
├── figure1_metrics_comparison.png
├── figure2_accuracy_vs_complexity.png
├── figure3_training_time.png
├── figure4_radar_chart.png
├── figure5_pareto_frontier.png
└── figure6_summary_table.png
```

---

## 📝 Esempio Tabella Risultati per Tesi

```
Model                  | MAE    | RMSE   | MAPE   | R²    | Training Time | Parameters
-----------------------|--------|--------|--------|-------|---------------|------------
LSTM                   | X.XXXX | X.XXXX | XX.XX% | X.XXX | XX.X min      | XX,XXX
LSTM Attention         | X.XXXX | X.XXXX | XX.XX% | X.XXX | XX.X min      | XX,XXX
LSTM Encoder-Decoder   | X.XXXX | X.XXXX | XX.XX% | X.XXX | XX.X min      | XX,XXX
LSTM Autoencoder       | X.XXXX | X.XXXX | XX.XX% | X.XXX | XX.X min      | XX,XXX
TimesNet               | X.XXXX | X.XXXX | XX.XX% | X.XXX | XX.X min      | XX,XXX
```

**Nota:** I valori effettivi verranno riempiti dopo gli esperimenti!

---

## 🎯 Consigli per Sezioni della Tesi

### Capitolo 1: Introduzione

**Cosa scrivere:**
- Importanza dell'efficienza energetica
- Smart buildings e IoT
- Ruolo del forecasting
- CityLearn Challenge

**Materiali da usare:**
- Statistiche su consumo energetico globale
- Paper su smart buildings
- Descrizione CityLearn Challenge

**Lunghezza:** 10-15 pagine

### Capitolo 2: Background

**Cosa scrivere:**
- RNN e problema vanishing gradient
- LSTM: equazioni complete
- Attention mechanisms
- TimesNet architecture

**Materiali da usare:**
- Paper originali (LSTM, Attention, TimesNet)
- Diagrammi architetture (puoi crearli con draw.io)
- Equazioni matematiche

**Lunghezza:** 25-30 pagine

### Capitolo 3: Metodologia

**Cosa scrivere:**
- Dataset CityLearn
- Preprocessing steps
- Architetture implementate (con diagrammi!)
- Hyperparameters
- Training strategy

**Materiali da usare:**
- Codice da `models/`
- Config da `configs/model_configs.yaml`
- Diagrammi di flusso

**Lunghezza:** 20-25 pagine

### Capitolo 4: Risultati

**Cosa scrivere:**
- Tabelle comparative
- Grafici performance
- Analisi per forecast horizon
- Ablation studies
- Statistical significance

**Materiali da usare:**
- Tutto da `thesis_results/`
- Figure da `figures/`
- Tabelle da `comparison_table.*`
- Report da `summary_report.txt`

**Lunghezza:** 30-40 pagine

### Capitolo 5: Discussione

**Cosa scrivere:**
- Interpretazione risultati
- Perché alcuni modelli funzionano meglio?
- Trade-offs identificati
- Limitations
- Confronto con letteratura

**Lunghezza:** 15-20 pagine

### Capitolo 6: Conclusioni

**Cosa scrivere:**
- Sintesi contributi
- Risposte alle research questions
- Lavoro futuro

**Lunghezza:** 8-10 pagine

---

## 🔬 Esperimenti Aggiuntivi (Opzionali ma Consigliati)

### 1. Ablation Study - Impact of Dropout

```bash
# Test different dropout values
for dropout in 0.1 0.2 0.3
do
    # Modifica configs/model_configs.yaml con dropout=$dropout
    python experiments/train.py --model lstm_attention --data your_data.csv
done
```

### 2. Ablation Study - Impact of Sequence Length

```bash
# Test different sequence lengths
for seq_len in 48 96 168
do
    # Modifica configs/model_configs.yaml con seq_len=$seq_len
    python experiments/train.py --model lstm_attention --data your_data.csv
done
```

### 3. Feature Engineering Study

Crea dataset con feature diverse:
- Baseline: solo consumption
- + time features (hour, day, month)
- + lag features (t-1, t-24, t-168)
- + rolling statistics (mean, std)

Train con ogni configurazione e confronta!

### 4. Multi-Building Analysis

```bash
# Run esperimenti su più building
for building in Building_1 Building_2 Building_3
do
    python experiments/run_all_experiments.py \
        --data data/raw/phase_1/${building}.csv \
        --output thesis_results/${building}
done
```

Poi analizza: quali building sono più difficili da predire?

---

## 📚 Bibliografia Consigliata

### Paper Fondamentali (da citare!)

**LSTM:**
- Hochreiter & Schmidhuber (1997). "Long short-term memory"

**Attention:**
- Bahdanau et al. (2015). "Neural Machine Translation by Jointly Learning to Align and Translate"
- Luong et al. (2015). "Effective Approaches to Attention-based Neural Machine Translation"

**TimesNet:**
- Wu et al. (2022). "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis"

**Energy Forecasting:**
- Cercare su Google Scholar: "building energy forecasting deep learning"
- "LSTM energy prediction"
- "smart building forecasting"

**CityLearn:**
- Vazquez-Canteli et al. (2019). "CityLearn: Diverse Gym Environments for RL in Cities"

### Dove Cercare Paper
- Google Scholar
- IEEE Xplore
- ArXiv.org
- ACM Digital Library

---

## 🎓 Criteri di Valutazione Tesi

### Cosa Valutano i Professori

1. **Chiarezza espositiva** (25%)
   - Tesi ben scritta e organizzata
   - Grafici e tabelle chiare
   - Spiegazioni comprensibili

2. **Rigore metodologico** (25%)
   - Esperimenti riproducibili
   - Statistical significance tests
   - Ablation studies

3. **Qualità risultati** (20%)
   - Esperimenti completi
   - Analisi approfondita
   - Confronti con state-of-the-art

4. **Contributo originale** (15%)
   - Cosa hai fatto di nuovo?
   - Insight interessanti
   - Framework riusabile

5. **Background teorico** (15%)
   - Comprensione profonda
   - Bibliografia completa
   - Related work esaustivo

---

## 🏆 Come Ottenere la Lode

### Elementi Che Fanno la Differenza

1. **✅ Esperimenti completi e rigorosi**
   - Tutti i modelli testati
   - Multiple seeds per riproducibilità
   - Statistical significance tests

2. **✅ Analisi critica eccellente**
   - Non solo "questi sono i risultati"
   - Ma "PERCHÉ questi risultati?"
   - Discussione approfondita

3. **✅ Codice professionale**
   - Ben documentato
   - Riproducibile
   - Repository GitHub pubblico

4. **✅ Presentazione chiara**
   - Grafici publication-quality
   - Tabelle ben formattate
   - Tesi ben scritta

5. **✅ Extra value**
   - Demo deployment (es. Streamlit app)
   - Paper submission a workshop
   - Contributo open-source

---

## 📞 Supporto e Debugging

### Problemi Comuni

**1. "Out of memory durante training"**
```bash
# Riduci batch size in configs/model_configs.yaml
batch_size: 32  # invece di 64
```

**2. "Training troppo lento"**
```bash
# Usa mixed precision
# In train.py, usa AdvancedTrainer con use_amp=True
```

**3. "Overfitting sui dati"**
```bash
# Aumenta regularization in configs/
dropout: 0.3  # invece di 0.2
weight_decay: 0.02  # invece di 0.01
```

**4. "Risultati non riproducibili"**
```python
# Fissa random seeds
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

---

## ✅ Final Checklist Prima della Consegna

### Codice
- [ ] Tutti i file committati su Git
- [ ] README aggiornato
- [ ] Requirements.txt completo
- [ ] Test funzionanti (`test_installation.py`)

### Esperimenti
- [ ] Tutti i modelli trainati
- [ ] Risultati salvati in `thesis_results/`
- [ ] Figure generate
- [ ] Tabelle create

### Tesi
- [ ] Tutti i capitoli scritti
- [ ] Figure inserite con caption
- [ ] Tabelle inserite
- [ ] Bibliografia completa (almeno 30 fonti)
- [ ] Revisione grammaticale
- [ ] Abstract scritto
- [ ] Ringraziamenti (se previsti)

### Presentazione
- [ ] Slide preparate (15-20 minuti)
- [ ] Demo funzionante (opzionale ma wow!)
- [ ] Provato la presentazione

---

## 🎉 Conclusione

**Hai tutto quello che ti serve per una tesi ECCELLENTE!**

📦 Progetto completo e funzionante
📊 Script automatici per esperimenti
📈 Generazione automatica figure
📚 Documentazione completa
🎓 Struttura tesi dettagliata

**Timeline consigliata: 6-8 settimane**

**Buona fortuna con la tesi! 🚀**
