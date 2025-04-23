import nltk # Needed to download 'stopwords' and 'wordnet'
nltk.download('stopwords') # Trenger bare å kjøre første gang
nltk.download('wordnet') # Trenger bare å kjære første gang
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
import re
from nltk.stem import WordNetLemmatizer
import pandas as pd

data = pd.read_csv("./Hotel_Reviews.csv")

def clean_sentences(data):
    all_text = []
    for sen in range(0, len(data)):
        # Remove all the special characters
        text = re.sub(r'\W', ' ', str(data.iloc[sen]))
        # Remove all numbers
        text = re.sub(r'\d', ' ', text)
        # remove all single characters
        text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
        # Remove single characters from the start
        text = re.sub(r'\^[a-zA-Z]\s+', ' ', text) 
        # Substituting multiple spaces with single space
        text = re.sub(r'\s+', ' ', text, flags=re.I)
        # Removing prefixed 'b'
        text = re.sub(r'^b\s+', '', text)
        # Lemmatization
        text = text.split()
        # Converting to Lowercase
        for c in range(len(text)):
            text[c] = text[c].lower()

        text = [WordNetLemmatizer().lemmatize(word) for word in text if word not in (stop_words)]
        text = ' '.join(text)

        all_text.append(text)
    return all_text

def combine_reviews(data):
    X_1, X_2 = clean_sentences(data["Negative_Review"]), clean_sentences(data["Positive_Review"])
    X = []
    labels_to_remove = []
    for n in range(len(X_1)):
        if X_1[n] in (None, "", 'negative') and X_2[n] in (None, "", 'positive'):
            labels_to_remove.append(n)
        else:
            if X_1[n] in (None, "", 'negative'):
                X.append(X_2[n])
            elif X_2[n] in (None, "", 'positive'):
                X.append(X_1[n])
            else:
                X.append(X_1[n] + " " + X_2[n])
    return pd.array(X), labels_to_remove

clean_data, labels = combine_reviews(data)