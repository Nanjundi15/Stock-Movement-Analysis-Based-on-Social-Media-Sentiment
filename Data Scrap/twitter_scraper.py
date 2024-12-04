import tweepy
import pandas as pd

# Replace with your own Twitter API Bearer Token
bearer_token = "AAAAAAAAAAAAAAAAAAAAALttxQEAAAAAOil5JpJ9Ssuo%2BZS8EpTYxTUTevc%3DWG5Vq3ma7nG8nZhADbeL9zGxUQISkFhiAETH0exuJKvR7MLeAh"

# Initialize the Client object for API v2
client = tweepy.Client(bearer_token=bearer_token)

# Function to fetch tweets
def fetch_tweets_v2(keyword, count=10):
    """
    Fetch recent tweets based on a keyword using Twitter API v2.
    Args:
        keyword (str): The search keyword.
        count (int): Number of tweets to fetch (max 100 per request).

    Returns:
        pd.DataFrame: DataFrame containing tweet text and creation time.
    """
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

# Main script
if __name__ == "__main__":
    # Specify the keyword and number of tweets
    keyword = "Python"  # Replace with your desired keyword
    count = 15         # Adjust the number of tweets to fetch

    # Fetch tweets and store them in a DataFrame
    tweets_df = fetch_tweets_v2(keyword, count)
    
    # Display the DataFrame
    print(tweets_df)
    
    # Save to CSV (optional)
    if not tweets_df.empty:
        tweets_df.to_csv("tweets.csv", index=False)
        print("Tweets saved to 'tweets.csv'.")
