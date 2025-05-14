# 📝 Text Summarization with Deep Learning

This project implements and compares multiple deep learning models (Simple RNN, GRU, LSTM, Attention, Scaled Dot-Product Attention) for **abstractive text summarization** using **samsum** dialogue datasets. It includes data preprocessing, training, evaluation, prediction, and a containerized setup using Docker.

---

## 📁 Project Structure
```
text-summarization/
├── data/
│   ├── train.csv
│   └── test.csv
├── models/
│   └── simple_rnn.py
├── training/
│   └── train_model.py
├── utils/
│   └── preprocessing.py
├── configs/
│   └── config.yaml
├── Dockerfile
├── docker-compose.yaml
├── requirements.txt
└── run_experiments.py

```

---

## 🚀 Features

- Supports five model types:
  - Simple RNN
  - GRU
  - LSTM
  - Basic Attention
  - Scaled Dot-Product Attention
- Dialogue summarization from CSV
- Tokenization using HuggingFace Transformers
- Clean separation of preprocessing, training, and evaluation
- Configurable training settings via YAML
- Dockerized workflow for consistency and portability

---

## 🛠️ Requirements

Make sure the following are installed:

- Docker
- Docker Compose

> You do **not** need Python installed locally. Everything runs inside Docker.

---

## 🐳 Getting Started with Docker

### Step 1: Build Docker Image

```bash
docker-compose build
docker-compose up
docker-compose run summarizer bash
```
To change which file runs, modify the command: field in docker-compose.yaml:

```bash
command: ["python", "run_experiments.py", "--config", "configs/config.yaml"]
```

## ⚙️ Configuration
All parameter settings are defined in:

configs/config.yaml as follow:
```bash
data:
  train_path: "data/train_dataset.csv"      # train data location
  val_path: "data/validation_dataset.csv"   # validation data location
  test_path: "data/test_dataset.csv"        # test data location

model:
  tokenizer_name: "t5-small"                # tokenizer model
  model_types:                              # defined models for comparison
    - "gru"
    - "lstm"
    - "basic_attention"
    - "scaled_attention"

training:
  max_input_len: 256                        # maximum input length
  max_output_len: 64                        # maximum output length
  epochs: 10                                # number of epoch

output:
  save_model_path: "saved_models/"          # define the output path

prediction:
  num_samples: 5                            # to produce summary for 5 samples by each model
```

## Dataeset Preparation

The Samsum dataset is a popular benchmark dataset in the field of Natural Language Processing (NLP), specifically used for training and evaluating models on the task of abstractive dialogue summarization.
Each entry in the dataset contains:

- dialogue: A short multi-turn conversation, typically between two people.
- summary: A brief summary of the entire conversation.

The dataset set is split into trainig, validation and testing in CSV files.
It contains:
- train: ~14,000 dialogue-summary pairs
- validation: ~800 pairs
- test: ~800 pairs

You can load and process the dataset explicitly running following script (optional)
``` 
python data/load_data.py
```

## 📈 Model Training, Evaluation and Prediction

This project performs a complete text summarization pipeline, including training, evaluation, and prediction across multiple models.

### 🚀 Main Tasks

1. **Train** each model on the training dataset and save the trained weights.
2. **Evaluate** all trained models on the test dataset and compare their performance.
3. **Generate Predictions** on sample test data to visualize the quality of each model's summaries.

### 🏃‍♂️ Run All Steps

You can run the full pipeline using:
``` 
python run_experiments.py 
``` 

> 💡 To skip a step (e.g., training or evaluation), simply **comment out the relevant code block** in `run_experiments.py`.

### 💾 Saved Models

After training, models are saved automatically to:

``` 
saved_models/<model_type>_model.h5
``` 

## 📊 Results

The models were evaluated on test accuracy, loss, and ROUGE scores to measure both token-level correctness and summarization quality.

| Model                | Test Accuracy | Test Loss | ROUGE-1    | ROUGE-2    | ROUGE-L    | ROUGE-Lsum |
| -------------------- | ------------- | --------- | ---------- | ---------- | ---------- | ---------- |
| Vanilla RNN          | 0.6929        | 1.917     | 0.1888     | 0.0256     | 0.1685     | 0.1679     |
| GRU                  | 0.7003        | 1.862     | 0.2215     | 0.0347     | 0.1965     | 0.1959     |
| LSTM                 | 0.6956        | 1.864     | 0.2111     | 0.0275     | 0.1875     | 0.1872     |
| Basic-Attention      | **0.7008**    | 1.931     | **0.2509** | **0.0406** | **0.2162** | **0.2155** |
| Scaled-dot Attention | 0.6804        | 2.276     | 0.2108     | 0.0350     | 0.1863     | 0.1855     |

### 🏁 Key Findings

- The current deep learning models may lack sufficient depth or complexity to fully capture the semantics of the input dialogue.

- With limited training data or shallow architectures, traditional extractive methods can still be competitive or even superior.

- It highlights the importance of model selection and tuning, especially when working with small datasets or resource-constrained environments.

## ✅ Bonus

As a baseline, a simple extractive summarization method using TF-IDF (Term Frequency-Inverse Document Frequency) is included.

This helps compare classical NLP techniques with deep learning-based approaches.

To run TF-IDF summarization on the test dialogues:

``` bash
python3 summarize_tfidf.py
``` 
The script generates summaries and computes ROUGE scores using the same test dataset.

TF-IDF Baseline ROUGE Scores:

``` yaml
ROUGE-1    : 0.2932
ROUGE-2    : 0.0937
ROUGE-L    : 0.2225
ROUGE-Lsum : 0.2493
``` 
Finally, the CSV file (train_dataset_with_tfidf_summary.csv) for the IF-IDF summarization results are created.

## 👩‍💻 Author
Cho Cho Mar
