import pandas as pd
from models.baseline_tfidf import extractive_summary_tfidf
from run_experiments import calculate_rouge_scores


df = pd.read_csv("data/test_dataset.csv")
# Apply summarization
df['tfidf_summary'] = df['dialogue'].apply(lambda x: extractive_summary_tfidf(x, num_sentences=3))

generated_summaries = df['tfidf_summary'].tolist()
true_summaries = df['summary']

rouge_results = calculate_rouge_scores(generated_summaries, true_summaries)
for key in rouge_results:
    print(f"{key} : {rouge_results[key]:.4f}")

# Optional: save to file
df.to_csv("train_dataset_with_tfidf_summary.csv", index=False)
