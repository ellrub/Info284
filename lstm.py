import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix

from prepare_data import clean_sentences, combine_reviews

# Load the CSV file
csv_data = 'Info284/data/DATASET_REDUX.csv' # Changed for each machine.
data = pd.read_csv(csv_data)

# Extract the relevant columns
# positive_reviews = data['Positive_Review'].tolist()
# negative_reviews = data['Negative_Review'].tolist()
# review_score = data['Reviewer_Score'].tolist()
positive_reviews = []
negative_reviews = []
neutral_reviews = []
for review in data:
    if data['Reviewer_Score'] >= 8:
        positive_reviews.append(data['Positive_Review'])
    elif data['Reviewer_Score'] >= 4:
        negative_reviews.append(data['Negative_Review'])
    else:
        neutral_reviews = data['Positive_Review'].tolist() + data['Negative_Review'].tolist()


# Combine the reviews and create labels
reviews = positive_reviews + negative_reviews + neutral_reviews
labels = [0] * len(negative_reviews) + [1] * len(neutral_reviews) + [2] * len(positive_reviews)

# Clean the reviews
cleaned_reviews = clean_sentences(pd.Series(reviews))

# Tokenize the data
tokenizer = Tokenizer(num_words = 5000)
tokenizer.fit_on_texts(cleaned_reviews)
sequences = tokenizer.texts_to_sequences(cleaned_reviews)

# Pad the sequences
maxlen = 100
data = pad_sequences(sequences, maxlen = maxlen)

# Convert labels to numpy array
labels = np.array(labels)

# Split the data into training and validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(Embedding(input_dim = 5000, output_dim = 128))
model.add(LSTM(units = 128, return_sequences = False))
model.add(Dropout(0.5))
model.add(Dense(units = 1, activation = 'sigmoid'))

# Compile the model
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs = 10, batch_size = 32, validation_data=(X_val, y_val))

# Save the model in the recommended Keras format
model.save('Info284/data/model/sentiment_lstm_model.keras')

print("Model training complete and saved as 'data/model/sentiment_lstm_model.keras'")

# Load the model
model = load_model('Info284\data\modelsentiment_lstm_model.keras')

# Evaluate the model on the validation set
val_predictions = model.predict(X_val)
val_predictions = (val_predictions > 0.5).astype(int)

# Print classification report and confusion matrix
print(classification_report(y_val, val_predictions))
print(confusion_matrix(y_val, val_predictions))

# Select a subset of the data for prediction
sample_reviews = X_val[:10]  # Select the first 10 reviews for prediction
sample_labels = y_val[:10]  # Corresponding labels

# Make predictions
predictions = model.predict(sample_reviews)

# Print predictions
for review, prediction, label in zip(sample_reviews, predictions, sample_labels):
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    actual_sentiment = "Positive" if label == 1 else "Negative"
    print(f"Review: {review} - Predicted Sentiment: {sentiment} - Actual Sentiment: {actual_sentiment}")