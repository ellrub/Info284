from sklearn.neighbors import KNeighborsClassifier
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

path1 = "./data/DATASET_REDUX.csv"
path2 = "data/Hotel_Reviews.csv"
df = read_csv(path2)

df = df[['Positive_Review', 'Negative_Review', 'Reviewer_Score']]

df['Sentiment'] = df['Reviewer_Score'].apply(lambda x: 'positive' if x >= 7 else 'negative')

df['Review'] = df['Positive_Review'] + ' ' + df['Negative_Review']

df.dropna(subset=['Review'], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(df['Review'], df['Sentiment'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_tfidf, y_train)

y_pred = knn.predict(X_test_tfidf)

train_accuracy = accuracy_score(y_train, knn.predict(X_train_tfidf))
test_accuracy = accuracy_score(y_test, y_pred)
overfitting = train_accuracy - test_accuracy

matrix = confusion_matrix(y_test, y_pred)

print("k = 5")
print(matrix)
print(f'train_accuracy: {train_accuracy}')
print(f'test_accuracy: {test_accuracy}')
print(f'overfitting: {overfitting}')
print("--------------------------------------------------")

plt.figure(figsize=(8, 6))
sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()











import pandas as pd
from gensim.models import Word2Vec
import psutil
import gc
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def do_data_chunked(
    _path,
    _preparation_mode="lemma",
    _fragment_channels=100,
    _vectorization_mode="skip",
    _pad_vectors=False,
    _chunksize=10000,  # Process data in chunks of 10,000 reviews
    _label_map = {"negative": 0, "neutral": 1, "positive": 2}
):
    """
    Processes hotel review data in chunks to handle large datasets.

    Args:
        _path (str): Path to the CSV file.
        _preparation_mode (str, optional): Text preparation mode. Defaults to "lemma".
        _fragment_channels (int, optional): Dimensionality of word vectors. Defaults to 100.
        _vectorization_mode (str, optional): Vectorization method ("skip" or "bow"). Defaults to "skip".
        _pad_vectors (bool, optional): Whether to pad vectors to a uniform length. Defaults to False.
        _chunksize (int, optional): Number of rows to process per chunk. Defaults to 10000.
        _label_map (dict): Maps text labels to numerical values

    Returns:
        tuple: (train_X, test_X, train_Y, test_Y, resolution_largest)
    """

    if _vectorization_mode not in ("skip", "bow"):
        raise ValueError("Invalid vectorization mode")

    _10_power_2 = 1024
    _python = psutil.Process()

    print(">>> memory usage: %.2f gb" % float(_python.memory_info().rss / (_10_power_2 * _10_power_2 * _10_power_2)))
    print("# # started preparation")

    _word2vec_model = None  # Initialize Word2Vec model

    all_train_X = []
    all_test_X = []
    all_train_Y = []
    all_test_Y = []
    _resolution_largest = 0

    for chunk in pd.read_csv(_path, usecols=["Negative_Review", "Positive_Review", "Reviewer_Score"], chunksize=_chunksize):
        print("    # processing chunk...")
        print("    >>> memory usage: %.2f gb" % float(_python.memory_info().rss / (_10_power_2 * _10_power_2 * _10_power_2)))

        _dataset = dataframe_convertion(chunk)  # Assuming this function works on a chunk
        _dataset, chunk_resolution_largest = dataset_preparation(_dataset, _mode=_preparation_mode)  # Assuming this works on a chunk
        _resolution_largest = max(_resolution_largest, chunk_resolution_largest)

        _dataset_X = dataset_extraction(_dataset, "review")  # Assuming this works on a chunk
        _dataset_Y = dataset_extraction(_dataset, "score")  # Assuming this works on a chunk

        print("    >>> memory usage: %.2f gb" % float(_python.memory_info().rss / (_10_power_2 * _10_power_2 * _10_power_2)))
        del _dataset  # Explicitly delete the chunk
        gc.collect()

        if _vectorization_mode == "skip":
            if _word2vec_model is None:
                _word2vec_model = Word2Vec(_dataset_X, min_count=1, vector_size=_fragment_channels, window=5, sg=1)
            else:
                _word2vec_model.build_vocab(_dataset_X, update=True)  # Update vocab with new chunk
                _word2vec_model.train(_dataset_X, total_examples=_word2vec_model.corpus_count, epochs=10) #train on the new chunk
        elif _vectorization_mode == "bow":
            if _word2vec_model is None:
                _word2vec_model = Word2Vec(_dataset_X, min_count=1, vector_size=_fragment_channels, window=5)
            else:
                _word2vec_model.build_vocab(_dataset_X, update=True)
                _word2vec_model.train(_dataset_X, total_examples=_word2vec_model.corpus_count, epochs=10)
        else:
            raise ValueError("provided vectorization mode does not exist")

        print("    >>> memory usage: %.2f gb" % float(_python.memory_info().rss / (_10_power_2 * _10_power_2 * _10_power_2)))
        # Vectorize and split the *current chunk* of data:
        _chunk_train_X, _chunk_test_X, _chunk_train_Y, _chunk_test_Y = dataset_vectorize_slash_split(
            _dataset_X, _dataset_Y, 0.2, _fragment_channels, _resolution_largest, len(_label_map), _word2vec_model
        )

        all_train_X.extend(_chunk_train_X)
        all_test_X.extend(_chunk_test_X)
        all_train_Y.extend(_chunk_train_Y)
        all_test_Y.extend(_chunk_test_Y)

        del _dataset_X, _dataset_Y
        gc.collect()
        print("    >>> memory usage: %.2f gb" % float(_python.memory_info().rss / (_10_power_2 * _10_power_2 * _10_power_2)))

    if not _pad_vectors:
        _resolution_largest = -1

    print("# # finished preparation")

    print("# # finished vectorization")
    print(">>> memory usage: %.2f gb" % float(_python.memory_info().rss / (_10_power_2 * _10_power_2 * _10_power_2)))
    return (
        np.array(all_train_X),  # Convert lists to NumPy arrays at the end
        np.array(all_test_X),
        np.array(all_train_Y),
        np.array(all_test_Y),
        _resolution_largest,
    )