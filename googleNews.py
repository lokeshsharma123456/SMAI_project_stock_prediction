import pandas as pd
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import yfinance as yf

# Set the date range for the data collection
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

# Set the ticker symbol for the company of interest
ticker_symbol = 'AAPL'

# Collect the news articles for the company of interest
url = f'https://news.google.com/rss/search?q={ticker_symbol}&hl=en-US&gl=US&ceid=US:en'
response = requests.get(url)
soup = BeautifulSoup(response.content, 'xml')
news_items = soup.findAll('item')

# Create an empty dataframe to store the data
df = pd.DataFrame(columns=['Date', 'News Headline', 'News Body', 'Sentiment Score', 'Closing Price', 'Next Day Prediction'])

# Collect the stock prices for the company of interest
stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)

# Initialize the sentiment analyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Loop through the news items and calculate the sentiment score
for news_item in news_items:
    news_date = datetime.strptime(news_item.pubDate.string.split(' GMT')[0], '%a, %d %b %Y %H:%M:%S')
    if news_date >= start_date and news_date <= end_date:
        news_headline = news_item.title.string
        news_body = news_item.description.string
        sentiment_score = sia.polarity_scores(news_body)['compound']
        closing_price = stock_data.loc[str(news_date.date())]['Close']
        next_day_prediction = stock_data.loc[str(news_date.date() + timedelta(days=1))]['Close']
        df = df.append({'Date': news_date, 'News Headline': news_headline, 'News Body': news_body,
                        'Sentiment Score': sentiment_score, 'Closing Price': closing_price, 'Next Day Prediction': next_day_prediction},
                       ignore_index=True)

# Save the data to a CSV file
df.to_csv('news_stock_data.csv', index=False)
