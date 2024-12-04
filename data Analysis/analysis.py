import tweepy
import pandas as pd
from textblob import TextBlob

# Replace with your own Twitter API Bearer Token
bearer_token = "AAAAAAAAAAAAAAAAAAAAALttxQEAAAAAOil5JpJ9Ssuo%2BZS8EpTYxTUTevc%3DWG5Vq3ma7nG8nZhADbeL9zGxUQISkFhiAETH0exuJKvR7MLeAh"

# Initialize the Client object for API v2
client = tweepy.Client(bearer_token=bearer_token)

# Function to fetch tweets
def fetch_tweets_v2(keyword, count=10):
    try:
        # Search tweets using the API v2 `search_recent_tweets` endpoint
        response = client.search_recent_tweets(
            query=keyword,
            tweet_fields=["created_at", "text"],
            max_results=min(count, 100)  # max_results must be ≤ 100
        )
        
        # Handle empty responses
        if not response.data:
            print("No tweets found for the given keyword.")
            return pd.DataFrame(columns=["text", "created_at"])
        
        # Extract relevant data
        data = [{"text": tweet.text, "created_at": tweet.created_at} for tweet in response.data]
        return pd.DataFrame(data)

    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame(columns=["text", "created_at"])

# Function to analyze sentiment
def analyze_sentiment(tweet_text):
    analysis = TextBlob(tweet_text)
    return analysis.sentiment.polarity  # Returns a value between -1 (negative) and 1 (positive)

# Main script
if __name__ == "__main__":
    # Specify the keyword and number of tweets
    keyword = "Python"  # Replace with your desired keyword
    count = 15          # Adjust the number of tweets to fetch

    # Fetch tweets and store them in a DataFrame
    tweets_df = fetch_tweets_v2(keyword, count)
    
    # Check if DataFrame is not empty
    if not tweets_df.empty:
        # Apply sentiment analysis to each tweet
        tweets_df['sentiment'] = tweets_df['text'].apply(analyze_sentiment)
        
        # Display the DataFrame with sentiment scores
        print(tweets_df[['text', 'sentiment']])
        
        # Save to CSV (optional)
        tweets_df.to_csv("tweets_with_sentiment.csv", index=False)
        print("Tweets with sentiment saved to 'tweets_with_sentiment.csv'.")
    else:
        print("No tweets found to analyze.")
