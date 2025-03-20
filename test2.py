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