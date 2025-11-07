# Struttura Tesi - CityLearn Forecasting

## 📚 Titolo Proposto

**"Deep Learning per il Forecasting del Consumo Energetico: Un Confronto tra Architetture LSTM e TimesNet per Smart Buildings"**

Oppure:

**"Predizione del Consumo Energetico in Smart Buildings mediante Tecniche di Deep Learning: Applicazione al CityLearn Challenge"**

---

## 📖 Struttura della Tesi (Consigliata)

### CAPITOLO 1: Introduzione (10-15 pagine)

#### 1.1 Motivazione e Contesto
- Importanza dell'efficienza energetica negli edifici
- Smart buildings e gestione intelligente dell'energia
- Ruolo del forecasting nel demand response
- CityLearn Challenge 2023: obiettivi e impatto

#### 1.2 Problema Affrontato
- Previsione del consumo energetico multi-step ahead
- Complessità dei pattern temporali (giornalieri, settimanali, stagionali)
- Variabili esogene (temperatura, radiazione solare, occupancy)
- Necessità di modelli accurati e real-time

#### 1.3 Obiettivi della Tesi
- **Obiettivo Principale:** Sviluppare e confrontare modelli di deep learning per forecasting energetico
- **Obiettivi Secondari:**
  1. Implementare 5 architetture state-of-the-art
  2. Confrontare performance su dataset CityLearn
  3. Analizzare trade-off accuracy/complessità
  4. Identificare best practices per evitare overfitting

#### 1.4 Contributi Originali
- Framework completo per forecasting energetico
- Confronto sistematico tra architetture LSTM e TimesNet
- Ottimizzazioni specifiche per time series energetiche
- Analisi delle tecniche anti-overfitting

#### 1.5 Organizzazione della Tesi
- Breve descrizione di ogni capitolo

---

### CAPITOLO 2: Background e Stato dell'Arte (25-30 pagine)

#### 2.1 Time Series Forecasting
- Definizione formale del problema
- Metodi tradizionali (ARIMA, SARIMA, Prophet)
- Limiti degli approcci statistici classici

#### 2.2 Deep Learning per Time Series
- Perché deep learning per forecasting?
- Vantaggi rispetto a metodi tradizionali
- Sfide specifiche (non-stationarity, long-term dependencies)

#### 2.3 Architetture LSTM
- **2.3.1 Recurrent Neural Networks (RNN)**
  - Problema del vanishing gradient
- **2.3.2 LSTM: Long Short-Term Memory**
  - Forget gate, input gate, output gate
  - Equazioni matematiche complete
  - Varianti (GRU, Bidirectional LSTM)
- **2.3.3 LSTM Autoencoder**
  - Encoder-decoder architecture
  - Applicazioni (anomaly detection, denoising)
- **2.3.4 Attention Mechanisms**
  - Bahdanau Attention (additive)
  - Luong Attention (multiplicative)
  - Self-Attention e Multi-Head Attention
- **2.3.5 LSTM Encoder-Decoder (Seq2Seq)**
  - Teacher forcing
  - Applications in multi-step forecasting

#### 2.4 TimesNet: State-of-the-Art
- Limitazioni dei modelli esistenti
- Temporal 2D-Variation Modeling
- Inception blocks per multi-scale features
- FFT per period detection
- Equazioni e architettura dettagliata

#### 2.5 Related Work
- Studi precedenti su energy forecasting
- Applicazioni di LSTM in smart buildings
- CityLearn Challenge: lavori correlati
- Tabella comparativa dei metodi esistenti

#### 2.6 Gap nella Letteratura
- Cosa manca negli studi precedenti
- Come questa tesi colma il gap

---

### CAPITOLO 3: Metodologia (20-25 pagine)

#### 3.1 Dataset: CityLearn Challenge 2023
- **3.1.1 Descrizione**
  - Phase 1 e Phase 2
  - Building profiles
  - Features disponibili (temperatura, solar generation, etc.)
- **3.1.2 Statistiche Descrittive**
  - Numero di samples, buildings
  - Range valori, distribuzione
  - Pattern temporali identificati
- **3.1.3 Preprocessing**
  - Normalizzazione (StandardScaler)
  - Gestione missing values
  - Outlier detection e treatment
  - Feature engineering (lags, rolling stats, time features)

#### 3.2 Problem Formulation
- Definizione formale del problema di forecasting
- Input: sequenza di lunghezza `seq_len=96`
- Output: predizione `pred_len=24` steps ahead
- Loss function: MSE
- Metriche di valutazione: MAE, RMSE, MAPE, R²

#### 3.3 Modelli Implementati

**3.3.1 LSTM Base**
```
Architettura:
- Input size: n_features
- Hidden layers: 2 x 128 units
- Dropout: 0.2
- Output: forecast_horizon
- Parametri: ~X
```

**3.3.2 LSTM Autoencoder**
- Encoder-decoder structure
- Latent representation
- Applicazione al forecasting

**3.3.3 LSTM Attention**
- Bahdanau attention mechanism
- Attention weights visualization
- Interpretability

**3.3.4 LSTM Encoder-Decoder**
- Sequence-to-sequence
- Teacher forcing ratio: 0.5
- Multi-step forecasting

**3.3.5 TimesNet**
- Period detection via FFT
- 2D convolutions
- Multi-period aggregation

#### 3.4 Training Strategy
- **3.4.1 Data Split**
  - Train: 70%, Validation: 15%, Test: 15%
  - Temporal split (no shuffling)
- **3.4.2 Hyperparameters**
  - Learning rate: 0.001
  - Batch size: 64
  - Optimizer: AdamW
  - Epochs: 100 (max)
- **3.4.3 Regularization**
  - Dropout: 0.1-0.3
  - Weight decay: 0.01
  - Gradient clipping: 1.0
  - Early stopping: patience 20
- **3.4.4 Advanced Techniques**
  - Mixed precision training
  - Learning rate scheduling (Cosine warmup)
  - Stochastic Weight Averaging

#### 3.5 Evaluation Metrics
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- R² (Coefficient of Determination)
- Horizon-specific metrics
- Computational cost (training time, inference time)

#### 3.6 Experimental Setup
- Hardware utilizzato (CPU/GPU)
- Software stack (PyTorch, Python, libraries)
- Riproducibilità (random seeds)

---

### CAPITOLO 4: Esperimenti e Risultati (30-40 pagine)

#### 4.1 Setup Sperimentale
- Configurazioni testate
- Grid search hyperparameters (se fatto)

#### 4.2 Risultati Quantitativi

**4.2.1 Performance Comparison**

**Tabella 1: Risultati su Test Set**
| Model | MAE | RMSE | MAPE | R² | Training Time | Inference Time | Parameters |
|-------|-----|------|------|-----|---------------|----------------|------------|
| LSTM | X.XX | X.XX | X.XX% | X.XX | X min | X ms | XXk |
| LSTM Autoencoder | X.XX | X.XX | X.XX% | X.XX | X min | X ms | XXk |
| LSTM Attention | X.XX | X.XX | X.XX% | X.XX | X min | X ms | XXk |
| LSTM Enc-Dec | X.XX | X.XX | X.XX% | X.XX | X min | X ms | XXk |
| TimesNet | X.XX | X.XX | X.XX% | X.XX | X min | X ms | XXk |

**4.2.2 Forecast Horizon Analysis**
- Grafici MAE/RMSE per ogni step (1-24)
- Come degrada performance all'aumentare dell'orizzonte

**4.2.3 Building-Specific Results**
- Performance per singolo building
- Quali building sono più difficili da predire

**4.2.4 Seasonal Analysis**
- Performance estate vs inverno
- Impact of temperature variations

#### 4.3 Risultati Qualitativi

**4.3.1 Visualization dei Forecast**
- Grafici predictions vs ground truth
- Esempi di giorni tipici (weekday, weekend)
- Casi di successo e failure cases

**4.3.2 Attention Weights Analysis**
- Heatmap attention weights
- Quali timesteps sono più importanti?
- Interpretabilità del modello

**4.3.3 Residual Analysis**
- Distribuzione errori
- Autocorrelazione residui
- Q-Q plots

#### 4.4 Ablation Studies

**4.4.1 Impact of Dropout**
- Performance con dropout 0.1, 0.2, 0.3
- Overfitting analysis

**4.4.2 Impact of Sequence Length**
- Test con seq_len = 48, 96, 168
- Trade-off memory/accuracy

**4.4.3 Impact of Feature Engineering**
- Baseline (solo consumption)
- + time features
- + lag features
- + rolling statistics
- Full feature set

**4.4.4 Impact of Advanced Training**
- Baseline optimizer (Adam)
- AdamW
- + Cosine scheduler
- + SWA
- + Mixed precision

#### 4.5 Model Comparison

**4.5.1 Accuracy vs Complexity**
- Scatter plot: MAE vs number of parameters
- Pareto frontier

**4.5.2 Speed vs Accuracy**
- Scatter plot: MAE vs inference time
- Best for real-time applications?

**4.5.3 Statistical Significance Tests**
- t-test o Wilcoxon test
- Sono le differenze statisticamente significative?

#### 4.6 Ensemble Methods
- Weighted ensemble results
- Improvement over single models

#### 4.7 Best Model Selection
- Quale modello scegliere?
- Raccomandazioni per diversi use cases

---

### CAPITOLO 5: Discussione (15-20 pagine)

#### 5.1 Interpretazione dei Risultati
- Perché TimesNet/Attention funziona meglio?
- Perché LSTM base è più veloce?
- Trade-offs identificati

#### 5.2 Implicazioni Pratiche
- Applicabilità in smart buildings reali
- Requisiti computazionali
- Integration con sistemi HVAC/BEMS

#### 5.3 Overfitting Analysis
- Train/val/test curves
- Tecniche che hanno funzionato meglio
- Lessons learned

#### 5.4 Limitations
- **5.4.1 Dataset Limitations**
  - Dimensione limitata
  - Mancanza di extreme events
- **5.4.2 Model Limitations**
  - Assunzioni fatte
  - Uncertainty quantification
- **5.4.3 Computational Constraints**
  - GPU memory limits
  - Real-time requirements

#### 5.5 Confronto con Letteratura
- Come si posizionano i risultati rispetto a state-of-the-art?
- Miglioramenti ottenuti

---

### CAPITOLO 6: Conclusioni e Lavoro Futuro (8-10 pagine)

#### 6.1 Sintesi dei Contributi
- Recap degli obiettivi raggiunti
- Contributi principali

#### 6.2 Conclusioni Principali
1. TimesNet è il migliore per accuracy (ma più lento)
2. LSTM Attention è il best trade-off
3. Feature engineering migliora del X%
4. Tecniche anti-overfitting essenziali

#### 6.3 Risposte alle Research Questions
- RQ1: Quale architettura è migliore?
- RQ2: Come prevenire overfitting?
- RQ3: Trade-off accuracy/speed?

#### 6.4 Lavoro Futuro

**6.4.1 Extensions**
- Uncertainty quantification (Bayesian deep learning)
- Multi-building joint forecasting
- Transfer learning tra buildings
- Online learning e adaptation

**6.4.2 Advanced Models**
- Transformers (Informer, Autoformer)
- Graph Neural Networks (spatial dependencies)
- Hybrid models (physics-informed neural networks)

**6.4.3 Applications**
- Integration con reinforcement learning per control
- Anomaly detection
- What-if scenario analysis

**6.4.4 Deployment**
- Model compression (pruning, quantization)
- Edge deployment
- API development

---

### APPENDICI

#### Appendice A: Codice Sorgente
- Link al repository GitHub
- Struttura del codice
- Come riprodurre gli esperimenti

#### Appendice B: Hyperparameters Completi
- Tabella con tutti gli hyperparameters testati
- Grid search results

#### Appendice C: Risultati Supplementari
- Tabelle dettagliate per ogni building
- Grafici aggiuntivi

#### Appendice D: Mathematical Formulations
- Derivazioni complete delle equazioni
- Loss functions
- Backpropagation through time

---

## 📊 Lunghezza Totale Stimata

- **Tesi Triennale:** 80-100 pagine
- **Tesi Magistrale:** 120-150 pagine
- **Con codice e appendici:** Può arrivare a 200+ pagine

---

## 🎯 Cosa Devi Fare Ora

### Priority 1: ESPERIMENTI (FONDAMENTALE)
```bash
# 1. Train tutti i modelli
for model in lstm lstm_attention lstm_encoder_decoder lstm_autoencoder timesnet
do
    python experiments/train.py --model $model --data data/citylearn/Building_1.csv
done

# 2. Evaluate tutti
for model in lstm lstm_attention lstm_encoder_decoder lstm_autoencoder timesnet
do
    python experiments/evaluate.py --model_path checkpoints/${model}_best.pth \
                                   --model_type $model \
                                   --data data/citylearn/Building_1.csv
done

# 3. Salva tutti i risultati in results/
```

### Priority 2: CREARE SCRIPT DI ANALISI
- Script per generare tutte le tabelle
- Script per generare tutti i grafici
- Statistical significance tests

### Priority 3: SCRIVERE LA TESI
- Usa LaTeX o Word
- Inizia da Background (Cap 2)
- Poi Methodology (Cap 3)
- Poi Results (Cap 4) - quando hai i dati
- Infine Introduction e Conclusions

---

## 📝 Template LaTeX Consigliato

```latex
\documentclass[12pt,a4paper,twoside]{report}
\usepackage[utf8]{inputenc}
\usepackage[italian]{babel}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{algorithm}
\usepackage{algorithmic}

\title{Deep Learning per il Forecasting del Consumo Energetico: \\
       Un Confronto tra Architetture LSTM e TimesNet}
\author{Il Tuo Nome}
\date{Anno Accademico 2024/2025}

\begin{document}
\maketitle
\tableofcontents

\chapter{Introduzione}
...
\end{document}
```

---

## ✅ Checklist Tesi

### Contenuto
- [ ] Background teorico completo
- [ ] Related work con almeno 30-40 paper
- [ ] Metodologia chiara e riproducibile
- [ ] Esperimenti su tutti i 5 modelli
- [ ] Tabelle di confronto
- [ ] Grafici di qualità pubblicazione
- [ ] Ablation studies
- [ ] Statistical tests
- [ ] Discussione critica
- [ ] Limitations oneste

### Forma
- [ ] Bibliografia completa (IEEE o ACM format)
- [ ] Figure con caption descrittive
- [ ] Tabelle formattate bene
- [ ] Equazioni numerate
- [ ] Codice ben commentato
- [ ] Repository GitHub pubblico
- [ ] README con istruzioni riproduzione

### Extra (per eccellere)
- [ ] Confronto con baseline (ARIMA, Prophet)
- [ ] Uncertainty quantification
- [ ] Interpretability analysis
- [ ] Deployment demo (Streamlit app)
- [ ] Paper submission (workshop o conference)

---

## 🏆 Suggerimenti per Tesi Eccellente

1. **Non limitarti a implementare**: Analizza PERCHÉ un modello funziona meglio
2. **Statistical rigor**: Usa test statistici, confidence intervals
3. **Ablation studies**: Dimostra l'impatto di ogni componente
4. **Visualizations**: Grafici belli e informativi
5. **Reproducibility**: Codice pulito, ben documentato, con seed fissi
6. **Critical thinking**: Discuti limitations onestamente
7. **Future work**: Proponi estensioni interessanti

---

## 💡 Valore Aggiunto per la Tesi

**Cosa rende questa tesi FORTE:**
1. ✅ Problema reale e attuale (energy efficiency)
2. ✅ 5 modelli state-of-the-art implementati
3. ✅ Framework completo e riusabile
4. ✅ Best practices per production
5. ✅ Code quality professionale
6. ✅ Documentazione completa

**Cosa aggiungerebbe EXTRA value:**
1. Confronto con metodi tradizionali (ARIMA)
2. Deployment di una demo interattiva
3. Pubblicazione codice open-source (già fatto!)
4. Paper submission a workshop
5. Contributo alla CityLearn community

---

## 🎓 Voto Stimato

Con questo lavoro, se fatto bene:
- **Triennale:** 100-110/110
- **Magistrale:** 105-110/110 con lode

**Fattori per la lode:**
- Esperimenti completi e approfonditi
- Analisi critica eccellente
- Codice professionale
- Presentazione chiara
- Contributo originale (es. nuova tecnica o insight)

---

## 📞 Prossimi Steps Immediati

1. **OGGI:** Carica dati CityLearn e fai primi training
2. **QUESTA SETTIMANA:** Completa tutti gli esperimenti
3. **SETTIMANA 2:** Analizza risultati e crea tabelle/grafici
4. **SETTIMANA 3-4:** Scrivi Capitolo 2 e 3
5. **SETTIMANA 5:** Scrivi Capitolo 4 (Results)
6. **SETTIMANA 6:** Scrivi Introduction e Conclusions
7. **SETTIMANA 7:** Revisione e polish

**Timeline totale stimata: 7-8 settimane per una tesi completa**
