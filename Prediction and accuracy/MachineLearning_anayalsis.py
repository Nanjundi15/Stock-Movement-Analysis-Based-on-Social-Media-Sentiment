import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from textblob import TextBlob

# 1. Preprocess Data
def preprocess_data(filename):
    """Load CSV and preprocess data."""
    try:
        data = pd.read_csv(filename)
        # Drop rows with missing data
        data.dropna(subset=['text'], inplace=True)
        
        # Add sentiment labels
        data['sentiment'] = data['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
        data['label'] = data['sentiment'].apply(lambda x: 1 if x > 0 else 0)
        
        return data[['text', 'label']]
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return pd.DataFrame()

# 2. Train Machine Learning Model
def train_model(data):
    """Train a machine learning model."""
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data['text'], data['label'], test_size=0.2, random_state=42
    )

    # Convert text to numerical format using CountVectorizer
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:\n", report)

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Plot prediction accuracy with points
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(y_test)), y_test, color="blue", label="Actual Labels")
    plt.scatter(range(len(y_pred)), y_pred, color="red", marker="x", label="Predicted Labels")
    plt.title("Prediction Accuracy")
    plt.xlabel("Data Point Index")
    plt.ylabel("Sentiment (0 = Negative, 1 = Positive)")
    plt.legend()
    plt.show()

    return model, vectorizer

# 3. Plot Sentiment Distribution
def plot_sentiment_distribution(data):
    """Plot the distribution of sentiments in the dataset."""
    sentiment_counts = data['label'].value_counts()
    plt.figure(figsize=(6, 4))
    sentiment_counts.plot(kind='bar', color=['red', 'blue'], alpha=0.7)
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.xticks(ticks=[0, 1], labels=["Negative", "Positive"], rotation=0)
    plt.show()

# 4. Make Predictions
def predict_sentiment(model, vectorizer, new_texts):
    """Predict sentiment for new texts."""
    new_texts_vec = vectorizer.transform(new_texts)
    predictions = model.predict(new_texts_vec)
    return predictions

# Main Script
if __name__ == "__main__":
    # Load and preprocess the dataset
    filename = "tweets_with_sentiment.csv"  # Replace with your file
    data = preprocess_data(filename)
    if data.empty:
        print("No data available for training.")
    else:
        print("Data Preprocessed Successfully!")

        # Plot sentiment distribution
        plot_sentiment_distribution(data)

        # Train model
        model, vectorizer = train_model(data)

        # Test predictions
        test_texts = ["Python is awesome!", "I hate coding bugs."]
        predictions = predict_sentiment(model, vectorizer, test_texts)

        # Display results
        for text, pred in zip(test_texts, predictions):
            sentiment = "Positive" if pred == 1 else "Negative"
            print(f"Tweet: '{text}' -> Sentiment: {sentiment}")

