# Architecture Decision Records

Project: Emotion Recognition from Text and Facial Images (HIT401 Capstone, Group 1)
Team: Md Ahnaf Azmain Fayek, Masafy Kamran, Nayem, Md Rakibul Hassan Emon
Supervisor: Thuseethan Selvarajah

Each record uses the same format: Status, Context, Decision, Alternatives considered, Consequences. All records have the status **Accepted**.

| ID | Decision |
|---|---|
| ADR-001 | Keep text and face as two separate channels |
| ADR-002 | Use MELD for text and FER2013 for faces |
| ADR-003 | Use fine-tuned DistilBERT as the main text model |
| ADR-004 | Use class weights, softened by a square root for text |
| ADR-005 | Use ResNet18 transfer learning for faces and test CBAM |
| ADR-006 | Drop the Vision Transformer |
| ADR-007 | Crop the face and combine the three face models with a soft vote |
| ADR-008 | Evaluate with macro F1 and repeated runs |
| ADR-009 | Test prompting only models on 200 sentences |
| ADR-010 | Build the app in Streamlit and run it from Google Colab |

---

## ADR-001: Keep text and face as two separate channels

**Status:** Accepted

**Context.** A person can write angry words and look calm, or smile and write something sad. Mixing both signals into one score hides this difference.

**Decision.** Text and face are trained, tested and reported separately. The app shows both answers side by side in two tabs. No score merges them.

**Alternatives considered.**
- Early or late fusion into one prediction. Rejected because it hides which signal said what, and the two datasets do not describe the same people.

**Consequences.**
- Each channel has its own test set and result.
- The app is simple to explain and to test.
- The project cannot say anything about how text and face agree for the same person.

---

## ADR-002: Use MELD for text and FER2013 for faces

**Status:** Accepted

**Context.** We needed public data with the same seven emotion labels so the two channels can be compared fairly.

**Decision.** MELD (9,989 training and 2,610 test lines) for text. FER2013 (28,709 training and 7,178 test images) for faces.

**Alternatives considered.**
- Single sentence emotion datasets. Rejected because MELD is a well known conversation benchmark with the same seven labels.
- Other face datasets such as in the wild collections. Rejected because FER2013 is free, small enough for Colab and widely compared.

**Consequences.**
- Both datasets are imbalanced (neutral is 47 percent of MELD, disgust is rare in both).
- MELD lines are used one at a time, without context.
- FER2013 images are 48 by 48 and have noisy labels. Human accuracy is about 65 percent.

---

## ADR-003: Use fine-tuned DistilBERT as the main text model

**Status:** Accepted

**Context.** We compared seven text models. The team had free GPU time on Colab and no paid language model service.

**Decision.** Fine-tune `distilbert-base-uncased` (2 epochs, batch 16, learning rate 2e-5, maximum length 64) with a weighted loss. Keep the bidirectional LSTM with GloVe and the TF-IDF models as comparison points.

**Alternatives considered.**
- Full BERT or RoBERTa. Rejected for training time and memory on free GPUs.
- Logistic Regression only. It gave 53.0 percent accuracy but a macro F1 of 0.24.
- Prompting only models. See ADR-009.

**Consequences.**
- DistilBERT reached 63.0 percent accuracy (mean of three runs) and macro F1 near 0.42.
- Training takes about two minutes.
- Disgust and fear stay weak (F1 near 0.1).

---

## ADR-004: Use class weights, softened by a square root for text

**Status:** Accepted

**Context.** Without weights the LSTM predicted only neutral and surprise. Both datasets have rare classes.

**Decision.** Use balanced class weights in the loss. For text, take the square root of the weights so rare classes get more attention without dominating the loss. Face models use the balanced weights as they are. The data itself is not changed.

**Alternatives considered.**
- Oversampling (SMOTE style). Rejected because copying very short sentences adds little.
- Focal loss. Not tried, because class weights were simpler to apply in both the PyTorch loop and the Hugging Face trainer.

**Consequences.**
- DistilBERT macro F1 rose from 0.39 to 0.43 with almost no change in accuracy (63.68 to 63.33 percent).
- The LSTM now predicts all seven classes, with accuracy 42.3 percent instead of 46.7 percent in that run.
- The square root factor was chosen while looking at test results, so figures may be slightly optimistic.

---

## ADR-005: Use ResNet18 transfer learning for faces and test CBAM

**Status:** Accepted

**Context.** FER2013 is small, so a model trained from scratch is limited. Published work suggests CBAM attention can add accuracy.

**Decision.** Compare three face models: a custom CNN from scratch, ResNet18 with ImageNet weights (224 by 224 input, Adam at 0.0001, 6 epochs), and the same ResNet18 with a CBAM block after each of its four stages.

**Alternatives considered.**
- Larger ResNets or VGG. Rejected for time and memory.
- Training from scratch only. Kept only as the baseline.

**Consequences.**
- ResNet18 reached about 65.2 percent, about 11 points above the custom CNN (54.4 percent).
- CBAM gave +2.10, +0.28 and −0.14 points in three runs, so no consistent gain. It stays in the app because it was best on our six cropped test photos, but we do not claim it is better.
- No data augmentation was used.

---

## ADR-006: Drop the Vision Transformer

**Status:** Accepted

**Context.** A Vision Transformer was listed as optional in the interim report.

**Decision.** Do not train a Vision Transformer. Spend the time on repeated runs and a careful test of CBAM.

**Alternatives considered.**
- Train a ViT. Rejected because plain ViTs need large data and FER2013 is small.

**Consequences.**
- The comparison covers three face models, not four.
- A ViT or hybrid model remains an open question.

---

## ADR-007: Crop the face and combine the three face models with a soft vote

**Status:** Accepted

**Context.** Whole photos with hair, shoulders and background gave wrong answers. Two smiling photos gave only 3 of 6 correct answers, and the three models disagreed.

**Decision.**
- Find the largest face with an OpenCV Haar cascade (`haarcascade_frontalface_default.xml`) and crop it with 10 percent padding.
- Combine the models by averaging their probability vectors: `(cnn_probs + resnet_probs + cbam_probs) / 3`.
- Show how many models agree with the final label.

**Alternatives considered.**
- No cropping. Rejected after the test above.
- Majority vote or using only the best model. Rejected because the soft vote uses the confidence of each model.

**Consequences.**
- After cropping, six photos gave 15 of 18 correct answers across the three models.
- A very confident wrong model can still win the vote.
- The soft vote has not been measured on the full FER2013 test set. The photo tests are small and not an independent sample.

---

## ADR-008: Evaluate with macro F1 and repeated runs

**Status:** Accepted

**Context.** Logistic Regression looked good at 53.0 percent but ignored most emotions. One CBAM run looked like a 2.10 point gain that did not repeat.

**Decision.** Report accuracy next to a majority baseline, macro F1, per-class F1 and confusion matrices. Train the main models more than once and report means and ranges. Keep the final notebook seeded (seed 42).

**Alternatives considered.**
- Accuracy only. Rejected because it rewards guessing the big class.

**Consequences.**
- Results are more honest, and the CBAM result was caught.
- The standard error of accuracy is about 1.1 points on FER2013 and 6.9 points on 200 lines, so small gaps are not reliable.
- There is no separate validation set and no paired significance test.

---

## ADR-009: Test prompting only models on 200 sentences

**Status:** Accepted

**Context.** BART zero-shot, Qwen2.5-1.5B-Instruct and TinyLlama-1.1B-Chat are slow on a free GPU and download about 7 GB.

**Decision.** Run them on the first 200 test lines with greedy decoding and at most 10 new tokens. If an answer has no valid label, use the first label (anger). Store the recorded results in the final notebook and keep the long run switched off by default (`RUN_LLM_EXPERIMENTS = False`).

**Alternatives considered.**
- Run all 2,610 lines. Rejected for time.
- Larger or paid models. Rejected because the project uses free, open tools only.

**Consequences.**
- Scores are 24.0 percent (BART), 50.0 percent (Qwen2.5) and 10.0 percent (TinyLlama).
- The estimate is wide, about ±6.9 points at 50 percent.
- The fallback rule penalises models that do not follow the answer format.

---

## ADR-010: Build the app in Streamlit and run it from Google Colab

**Status:** Accepted

**Context.** The team needed a working demonstration without paid hosting.

**Decision.** Build one Streamlit app with two tabs (text and face). The final notebook writes `app.py`, starts it in Colab and opens a public `trycloudflare.com` link with cloudflared. Trained models are saved to Google Drive (`emotion_project`) so they do not need training again.

**Alternatives considered.**
- A hosted web service. Rejected because of cost.
- Separate apps for each channel. Rejected because the project needs one application.

**Consequences.**
- The app runs only while the Colab session is open.
- Some campus wifi networks block the tunnel, so a phone hotspot or mobile data is used.
- The final app code is in `04_App_Final_Emotion_Recognition.ipynb`, Section 3.
