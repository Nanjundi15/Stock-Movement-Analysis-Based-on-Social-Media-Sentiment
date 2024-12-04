import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Load the DataFrame from the saved CSV file
filename = 'tweets_with_sentiment.csv'  # Replace with your actual CSV file
try:
    tweets_df = pd.read_csv(filename)
    print(f"Successfully loaded data from '{filename}'")
except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
    exit()

# Function to analyze sentiment
def analyze_sentiment(text):
    """Analyze sentiment polarity of a text."""
    blob = TextBlob(text)
    return blob.sentiment.polarity  # Sentiment polarity: -1 (negative) to +1 (positive)

# If sentiment column is missing, calculate it
if 'sentiment' not in tweets_df.columns:
    print("Sentiment column not found. Calculating sentiment scores...")
    tweets_df['sentiment'] = tweets_df['text'].apply(analyze_sentiment)

# Summary statistics of sentiment scores
sentiment_summary = tweets_df['sentiment'].describe()
print("\nSummary Statistics for Sentiment Scores:")
print(sentiment_summary)

# Plot histogram of sentiment scores
plt.figure(figsize=(8, 6))
plt.hist(tweets_df['sentiment'], bins=20, color='blue', alpha=0.7, edgecolor='black')
plt.title("Sentiment Distribution of Tweets")
plt.xlabel("Sentiment Score")
plt.ylabel("Frequency")
plt.grid(axis='y', alpha=0.75)
plt.show()
