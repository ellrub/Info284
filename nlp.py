import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from prepare_data import clean_sentences

dataset = pd.read_csv("Info284\data\Hotel_Reviews.csv")
y_n = dataset["Reviewer_Score"].values
y = []
for value in y_n:
    if value < 4:
        y.append("negative")
    elif value < 8:
        y.append("neutral")
    else:
        y.append("positive")

X_1, X_2 = dataset["Negative_Review"], dataset["Positive_Review"]
X_1, X_2 = clean_sentences(X_1), clean_sentences(X_2)
X = []
for n in range(len(X_1)):
    X.append(X_1[n] + " " + X_2[n])

cv = CountVectorizer()
X = cv.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

bnb = BernoulliNB(binarize=0)
model = bnb.fit(X_train, y_train)
y_pred = bnb.predict(X_test)
print("BernoulliNB: ")
print(classification_report(y_test, y_pred), end="\n")

print("MultinomialNB: ")
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test) 
clf_report = classification_report(y_test, y_pred, output_dict=True)

sb.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True)
plt.show()