# Design: MultiBranchNet — CNN multi-input per il riconoscimento di strumenti su IRMAS

**Data:** 2026-07-04
**Stato:** approvato a voce, in attesa di review scritta

## Contesto e obiettivo

Il progetto attuale fa finetuning di PANNs CNN14 (input singolo: log-mel in-graph).
Si sostituisce con un'architettura **multi-input progettata ad hoc** che tratta le
rappresentazioni tempo-frequenza come immagini e le dà in pasto a backbone CNN
pre-addestrate su ImageNet, più rami custom addestrati da zero.

Motivazione: progetto universitario — il valore da dimostrare è (a) un'architettura
originale, (b) una pipeline completa con preprocessing esplicito e riproducibile,
(c) un'analisi sperimentale (ablation study per ramo). Le metriche assolute sono
secondarie.

Vincoli:
- Training su **Google Colab gratuito (GPU T4)**, sessioni ~3-4 h.
- Dataset: IRMAS (6.705 clip di training mono-strumento da 3 s, 11 classi;
  test polifonico multi-label a lunghezza variabile).
- Protocollo di valutazione IRMAS ufficiale invariato.

## Architettura

```
mel    (1×128×T) ──▶ ResNet18 pre-addestrata ImageNet ──▶ 512-d ─┐
CQT    (1×84×T)  ──▶ ResNet18 pre-addestrata ImageNet ──▶ 512-d ─┼▶ concat ─▶ MLP ─▶ 11 logit
wave   (66150,)  ──▶ Conv1D custom (da zero)           ──▶ 256-d ─┤
chroma (1×12×T)  ──▶ mini-CNN 2D (da zero)             ──▶ 128-d ─┘
```

- **Rami mel e CQT**: ResNet18 di torchvision con pesi ImageNet. L'input mono-canale
  viene **replicato 1→3 canali** per riusare intatta la prima conv; una BatchNorm2d
  d'ingresso per ramo adatta le statistiche degli spettrogrammi a quelle attese.
  Embedding: uscita del global average pooling (512-d), testa fc originale rimossa.
- **Ramo waveform**: ~5 blocchi Conv1D (conv + BN + ReLU + pooling progressivo) →
  global pooling → 256-d. Addestrato da zero. È il ramo più "originale" ma con
  aspettative di contributo basse — da verificare nell'ablation.
- **Ramo chroma**: 3 blocchi conv 2D piccoli → 128-d. Ipotesi dichiarata: ridondante
  rispetto alla CQT (chroma = CQT ripiegata su 12 classi di altezza; codifica
  l'armonia, non il timbro). L'ablation la conferma o smentisce.
- **Fusion**: concatenazione degli embedding dei soli rami attivi → MLP
  (input dinamico → 512 → 11 logit) con dropout.
- **Ogni ramo è attivabile/disattivabile da config** (`branches: {mel: true, cqt: true,
  wave: true, chroma: true}`). La testa calcola la dimensione d'ingresso dai rami attivi.
  Almeno un ramo deve essere attivo.
- Loss: `BCEWithLogitsLoss` (multi-label, coerente col test IRMAS); sigmoid a inferenza.

## Preprocessing e dati

Nuovo modulo `features.py` (funzioni pure di estrazione, condivise tra preprocessing
e valutazione) + nuovo script `scripts/preprocess.py`.

Parametri feature (tutti nel config):
- Sample rate **22.050 Hz**, mono (sostituisce i 32 kHz richiesti da PANNs).
- **Log-mel**: 128 bande, n_fft 2048, hop 512 → `128 × ~130` per clip da 3 s.
- **CQT**: 84 bin (7 ottave × 12), hop 512, in dB → `84 × ~130`.
- **Chroma**: 12 bin, hop 512 → `12 × ~130`.
- **Waveform**: 66.150 campioni grezzi (3 s), pad/trim come oggi.

Flusso:
1. `scripts/preprocess.py` legge `data/IRMAS-TrainingData/**/*.wav`, calcola le 4
   rappresentazioni e salva un `.npz` **float16** per clip (~190 KB → ~1,3 GB totali)
   in `data/features/train/`, preservando la struttura cartella-per-classe.
2. Calcola e salva le **statistiche di normalizzazione (media/std per feature) sul solo
   training set** (`data/features/stats.json`); train e test le riusano.
3. Il nuovo `IRMASFeaturesDataset` legge gli `.npz` (lazy), normalizza e restituisce
   `dict` di tensori + target multi-hot.
4. **Il test set non viene precalcolato**: in valutazione le feature si calcolano al
   volo per finestra con le stesse funzioni di `features.py` (stessi parametri, stesse
   statistiche), garantendo coerenza train/test. Estrazione: librosa.

## Training

Struttura a due fasi conservata dall'attuale `train.py`:
1. **Warmup**: ResNet18 congelate; si addestrano testa MLP + rami custom (wave, chroma).
2. **Finetuning**: si scongela tutto; LR discriminativi (backbone basso ~5e-5, resto
   ~5e-4), cosine decay, early stopping su micro-F1 di validazione, checkpoint del best.

Augmentation (ripensata per feature precalcolate):
- **SpecAugment** (mascheramento tempo/frequenza) su mel e CQT, solo in training,
  implementato come transform sul tensore precalcolato.
- **Mixup** a livello di batch, stesso λ applicato coerentemente a tutti gli input
  attivi e ai target.
- Gain/rumore sulla waveform: rimossi (non significativi su feature log precalcolate).

Difese anti-overfitting (rischio principale: 6.705 clip, 2 backbone da 11 M parametri):
warmup a backbone congelate, weight decay, dropout nella testa, SpecAugment, mixup,
early stopping. Opzione di riserva documentata: scongelare solo layer3+layer4 delle ResNet.

Split train/val stratificato (15 %) e seed globale: invariati.

## Valutazione e ablation study

- **Protocollo IRMAS invariato**: sliding window 3 s / hop 1 s sulle clip di test
  polifoniche, aggregazione `mean` delle sigmoid per clip, soglia ottimizzata sulla
  validazione, micro/macro precision/recall/F1. Si riusano `windowing.py` e
  `metrics.py`; a `metrics.py` si aggiunge il **F1 per classe**.
- **Ablation study**: run multiple che differiscono *solo* per `branches` —
  `mel` → `mel+cqt` → `mel+cqt+wave` → `mel+cqt+wave+chroma` (stesso seed, epoche,
  split, procedura di soglia). Output: tabella comparativa in `outputs/ablation.md`.
  Uno script/config-set dedicato rende le run ripetibili.
- Punto dichiarato nella relazione (non nascosto): la soglia è ottimizzata su clip
  mono-strumento ma il test è polifonico — discrepanza di distribuzione del protocollo
  IRMAS. Esperimento accessorio suggerito: soglia fissa 0.5 vs ottimizzata.
- Figura bonus per la relazione: t-SNE/UMAP degli embedding fusi colorata per classe.

## Impatto sul codice esistente

| Invariato | Sostituito/rimosso | Nuovo |
|---|---|---|
| `labels.py`, `metrics.py` (+ per-class F1), `windowing.py`, `utils.py`, `scripts/download_data.py`, protocollo test, split stratificato | `models/cnn14.py` → `models/multibranch.py`; `scripts/download_pretrained.py` rimosso (torchvision scarica ResNet18 da sé); `data/dataset.py` riscritto sugli `.npz`; `data/transforms.py` → SpecAugment + mixup multi-input | `features.py`; `scripts/preprocess.py`; sezione `branches:` e `features:` nel config; `IRMASFeaturesDataset`; test pytest nuovi; notebook Colab aggiornato |

Dipendenze: si aggiungono `librosa` e `torchvision`; si rimuove `torchlibrosa`.

## Testing

- Test esistenti che restano validi: labels, metrics, windowing.
- Nuovi test: shape e valori finiti delle feature (`features.py`); coerenza
  preprocessing/eval (stessa clip → stesse feature per le due vie); forward del
  modello con **ogni combinazione di rami** (dimensione testa dinamica); mixup
  multi-input; overfit sanity check end-to-end su un mini-batch.

## Fuori scopo

- Ricerca di iperparametri estensiva; architetture attention/transformer;
  addestramento del vecchio CNN14 come baseline (il confronto in relazione può
  citare i numeri di letteratura); deployment/inferenza real-time.
