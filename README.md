### **README - Stock Movement Analysis Based on Social Media Sentiment**

#### **Objective**
The goal of this project is to analyze the relationship between social media sentiment and stock market movement. We collect tweets about a specific stock, analyze their sentiment using natural language processing (NLP), and assess whether sentiment correlates with stock price movements. This can potentially be used for predicting stock movements based on public sentiment.

---

### **Prerequisites**
1. **Python** installed on your system.
2. Install required libraries:
   ```bash
   pip install pandas textblob tweepy scikit-learn matplotlib seaborn
   ```
3. **Twitter Developer Account**:
   - You will need access to the Twitter API via your [Twitter Developer account](https://developer.twitter.com/en/apps) and generate API keys (Bearer Token).
   
4. **Stock Data**:
   - A dataset containing historical stock prices (e.g., CSV file with date and closing price).
   - Alternatively, you can use an API like Yahoo Finance or Alpha Vantage to fetch stock data.

---

### **Steps and Workflow**

1. **Data Collection**:
   - The script collects recent tweets mentioning a stock using the Twitter API (via `tweepy`).
   - Tweets are fetched using specific stock-related keywords.
   - The **sentiment of the tweets** is analyzed using **TextBlob**. Sentiment polarity values are assigned to each tweet:
     - **Positive sentiment**: Sentiment value greater than 0.
     - **Negative sentiment**: Sentiment value less than or equal to 0.

2. **Sentiment Analysis**:
   - **TextBlob** is used to analyze the sentiment of each tweet.
   - Sentiment scores are calculated, which are used to categorize the tweets as positive or negative.

3. **Modeling**:
   - **Logistic Regression** is used to predict the sentiment of tweets.
   - The dataset is split into training and testing sets, with features extracted from the tweet text using `CountVectorizer`.
   - The model is trained, evaluated, and tested for sentiment prediction.
   
4. **Stock Price Data**:
   - Historical stock prices (e.g., stock opening or closing price) are collected.
   - The stock price movements are compared against the sentiment of tweets on specific days to analyze correlations.

5. **Sentiment and Stock Movement Analysis**:
   - Visualizations are generated to show the sentiment distribution of tweets.
   - A **confusion matrix** and **accuracy metrics** are plotted to evaluate the sentiment prediction model.
   - A **scatter plot** compares the actual sentiment with predicted sentiment for better visualization.
   - **Stock movement** is then analyzed in conjunction with sentiment analysis, using graphs to compare the changes in stock prices with the sentiments extracted from social media.

---

### **How to Use**
1. **Download the Twitter Dataset**:
   - Use the `fetch_tweets_v2()` function to collect tweets about the stock symbol you are interested in (e.g., Apple or Tesla).
   - Pass the stock keyword as a parameter (e.g., `"Apple stock"`).

2. **Collect Stock Data**:
   - You can download stock data from sources like Yahoo Finance or use an API such as Alpha Vantage to collect stock prices.
   - Ensure you have the correct stock data (e.g., date and closing price) and merge it with your tweet sentiment data.

3. **Preprocess Data**:
   - The script preprocesses tweet data to generate sentiment labels (1 for positive sentiment and 0 for negative sentiment).
   - Sentiment scores are calculated using **TextBlob**.

4. **Train and Evaluate the Model**:
   - The model is trained using the **Logistic Regression** algorithm to predict tweet sentiment.
   - Model evaluation is performed with accuracy, classification report, and a confusion matrix.
   
5. **Visualize Results**:
   - The sentiment distribution of tweets is shown using histograms.
   - A **confusion matrix** is plotted to visualize the performance of the sentiment prediction model.
   - A **scatter plot** of predicted vs. actual sentiments is generated.
   - The stock movement is analyzed alongside sentiment changes using stock price data.

---

### **Folder Structure**
```
Stock-Sentiment-Analysis/
│
├── data/                       # Folder containing datasets
│   ├── tweets_with_sentiment.csv  # CSV file with tweet text and sentiment
│   ├── stock_data.csv            # CSV file with historical stock prices
│
├── src/                        # Folder containing scripts
│   ├── sentiment_analysis.py     # Script for sentiment analysis and model training
│   ├── fetch_tweets.py           # Script to fetch tweets using Twitter API
│   └── stock_analysis.py         # Script for stock analysis and data processing
│
└── README.md                   # Project documentation (this file)
```

---

### **Customization**
1. **Change Stock Symbol**: Modify the keyword used in the `fetch_tweets_v2()` function to reflect the stock you are interested in.
2. **Modify Sentiment Labeling**: You can adjust the sentiment thresholds or use a different model (e.g., Naive Bayes, SVM) for sentiment classification.
3. **Stock Data Sources**: Fetch stock data from APIs like Yahoo Finance, Alpha Vantage, or Quandl if you don’t have a CSV file.
4. **Model Tuning**: You can improve the model by adding more data, fine-tuning parameters, or using deep learning models.

---

### **Output**
1. **Sentiment Analysis Output**:
   - **Confusion Matrix**: A heatmap showing the true vs. predicted sentiments.
   - **Prediction Accuracy**: A scatter plot comparing actual and predicted sentiment values.
   - **Sentiment Distribution**: A histogram showing the distribution of tweet sentiments (positive and negative).

2. **Stock Movement Analysis Output**:
   - Correlation between tweet sentiment and stock price movement.
   - Visualization comparing stock price changes with sentiment data.
   
3. **Console Output**:
   - Accuracy and classification report for sentiment prediction.
   - Sentiment prediction results for test text inputs.

---

### **Future Improvements**
- **Advanced Sentiment Analysis**: Implement more advanced NLP techniques such as BERT or GPT-3 for sentiment analysis.
- **Deep Learning Models**: Experiment with LSTM or BERT models for better sentiment classification.
- **Stock Price Prediction**: Extend the project to predict future stock price movements using sentiment data combined with historical stock prices.

---
