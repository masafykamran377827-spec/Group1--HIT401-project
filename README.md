# Emotion Recognition from Text and Facial Images

HIT401 Capstone Project, Charles Darwin University (Faculty of Science and Technology), Group 1.
Supervisor: Thuseethan Selvarajah.

| Student | ID |
|---|---|
| Md Ahnaf Azmain Fayek | S374489 |
| Masafy Kamran | S377827 |
| Nayem | S375296 |
| Md Rakibul Hassan Emon | S375332 |

## What this project does

Two separate emotion recognition systems, one for text and one for faces. They are never merged into one score. One Streamlit application shows both, in two tabs.

- **Text channel:** MELD dialogue lines, 7 emotions (anger, disgust, fear, joy, neutral, sadness, surprise).
- **Face channel:** FER2013 grayscale face photos, 7 emotions (angry, disgust, fear, happy, neutral, sad, surprise).

## Where is the app code?

The final application code is in **`04_App_Final_Emotion_Recognition.ipynb`**. Section 3 of that notebook ("Final application") writes `app.py` (the two tab Streamlit app) and starts it with a public link. The earlier notebooks 02 and 03 each contain a smaller one tab app that was used during development.

## Repository files

| File | Purpose |
|---|---|
| `01_MELD_Data_Exploration.ipynb` | Loads MELD, checks class counts, trains Logistic Regression and a feedforward network as baselines. |
| `02_MELD_Text_Models_LSTM_DistilBERT_LLM.ipynb` | Bidirectional LSTM with GloVe, DistilBERT, and prompting only models (BART zero-shot, Qwen2.5-1.5B, TinyLlama-1.1B). Also has the first text app. |
| `03_FER2013_Face_Models_CNN_ResNet_CBAM.ipynb` | Custom CNN, ResNet18 (ImageNet weights) and ResNet18 with CBAM attention on FER2013. Also has the first face app. |
| `04_App_Final_Emotion_Recognition.ipynb` | **Final notebook.** Trains or loads every model, prints all results, and runs the final two tab app (`app.py`). |

## Models

| Channel | Models |
|---|---|
| Text | Logistic Regression, Feedforward NN (TF-IDF); bidirectional LSTM (GloVe 100d); DistilBERT; BART zero-shot; Qwen2.5-1.5B-Instruct; TinyLlama-1.1B-Chat |
| Face | Custom CNN; ResNet18; ResNet18 + CBAM. Final answer in the app is the soft vote (mean of the three probability vectors) |

The face tab first finds the largest face with an OpenCV Haar cascade, crops it with 10 percent padding, and then runs the three models.

## Results (official test sets)

| Model | Test accuracy |
|---|---|
| Always neutral (MELD baseline) | 48.1% |
| Logistic Regression | 53.0% |
| Feedforward NN | 48.5% |
| Bidirectional LSTM (mean of 3 runs) | 44.1% |
| **DistilBERT (mean of 3 runs)** | **63.0%** |
| BART zero-shot (200 lines) | 24.0% |
| Qwen2.5-1.5B (200 lines) | 50.0% |
| TinyLlama-1.1B (200 lines) | 10.0% |
| Always happy (FER2013 baseline) | 24.7% |
| Custom CNN (mean of 4 runs) | 54.4% |
| ResNet18 (mean of 4 runs) | 65.2% |
| ResNet18 + CBAM (mean of 3 runs) | 65.7% |

Notes: accuracy has a standard error of about 1.1 points on FER2013 and 6.9 points on the 200 line tests. Class weights are used in every trained model. CBAM did not give a consistent gain over ResNet18 (+2.10, +0.28, −0.14 points in three runs).

## How to run

1. Open `04_App_Final_Emotion_Recognition.ipynb` in Google Colab and choose a GPU runtime (Runtime, Change runtime type).
2. Run the first cell to install packages (`transformers`, `accelerate`, `gensim`, `streamlit`, `datasets==2.19.0`).
3. Run the setup cell. It mounts Google Drive and saves everything in `MyDrive/emotion_project`.
4. Run the text and image sections to train. Saved models are reused on the next run. To retrain, set `FORCE_RETRAIN_TEXT = True` and `FORCE_RETRAIN_IMAGE = True` in the setup cell.
5. The prompting experiments (BART, Qwen2.5, TinyLlama) are off by default because they download about 7 GB. Set `RUN_LLM_EXPERIMENTS = True` to run them. Their recorded results are already stored in the notebook.
6. Run Section 3. The first cell writes `app.py`. The second cell starts Streamlit and prints a `trycloudflare.com` link. Open that link to use the app.

If the link cannot be reached, wait 30 seconds and reload. Some campus wifi networks block it, so use a phone hotspot or mobile data.

## Settings

- Random seed: 42 (final notebook).
- LSTM: 2 bidirectional layers, 256 hidden units, vocabulary 8000, length 30, Adam at 0.001, 15 epochs, batch 64.
- DistilBERT: `distilbert-base-uncased`, 2 epochs, batch 16, learning rate 2e-5, maximum length 64.
- Face CNN: Adam at 0.001, 15 epochs. ResNet18 and CBAM: Adam at 0.0001, 6 epochs, 224 by 224 input.
- Class weights: balanced, with a square root softening for text.

## Data sources

- MELD: https://github.com/declare-lab/MELD (cloned automatically by the notebook).
- FER2013: Hugging Face dataset `Jeneral/fer-2013` (loaded automatically).
- GloVe vectors: `glove-wiki-gigaword-100` through gensim.

## Notes and limits

- MELD lines are read one at a time, without dialogue context.
- No separate validation set was used, and there is no data augmentation.
- The app is a research prototype. It is not intended for decisions about people.
