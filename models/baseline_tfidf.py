
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
import numpy as np
import nltk
nltk.download('punkt_tab')

def extractive_summary_tfidf(text, num_sentences=1):

    """
    Generate an extractive summary by selecting the top-ranked sentences based on TF-IDF scores.

    Parameters:
    -----------
    text : str
        The input text to summarize.
    num_sentences : int, optional (default=3)
        The number of sentences to include in the summary.

    Returns:
    --------
    summary : str
        A summary composed of the top `num_sentences` ranked sentences from the input text.
        If the input has fewer than `num_sentences` sentences, the original text is returned.
    """

    # Step 1: Split the text into individual sentences
    if not isinstance(text, str):
        return ""
    
    sentences = sent_tokenize(text)

    # Step 2: Create TF-IDF vectors for each sentence
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)  # shape: (n_sentences, n_terms)

    # Step 3: Compute a score for each sentence (sum of TF-IDF weights)
    sentence_scores = np.asarray(tfidf_matrix.sum(axis=1)).flatten()

    # Step 4: Get indices of top 'num_sentences' scored sentences
    top_indices = sentence_scores.argsort()[::-1][:num_sentences]

    # Step 5: Sort top sentences by their original order in text
    top_indices_sorted = sorted(top_indices)

    # Step 6: Combine and return the summary
    summary = ' '.join([sentences[i] for i in top_indices_sorted])
    return summary