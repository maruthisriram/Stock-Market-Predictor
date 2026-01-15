import yfinance as yf
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta

def fetch_historical_data(symbol, period="1mo", interval="1d"):
    """
    Fetch historical stock data using yfinance.
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            return None
        return df
    except Exception as e:
        print(f"Error fetching historical data for {symbol}: {e}")
        return None

def fetch_news_sentiment(symbol):
    """
    Fetch recent news using yfinance and calculate average sentiment score.
    """
    try:
        print(f"Fetching news for {symbol}...")
        ticker = yf.Ticker(symbol)
        news = getattr(ticker, 'news', [])
        
        # Debugging news structure
        if not news:
            print(f"No news found for {symbol} using yfinance.")
            return 0, []

        analyzer = SentimentIntensityAnalyzer()
        sentiments = []
        news_list = []

        for article in news:
            # Check for multiple levels of nesting
            content = article.get('content', {})
            # If no content dict, use the article itself
            data_source = content if content else article

            # Title extraction - highly robust
            title = data_source.get('title') or data_source.get('headline') or article.get('title') or article.get('headline')
            if not title:
                continue
                
            # Publisher extraction
            publisher = data_source.get('publisher') or data_source.get('source') or article.get('publisher') or article.get('source') or "Market News"
            
            # Link extraction - prioritize the actual article URL
            link = data_source.get('link') or data_source.get('url') or data_source.get('previewUrl') or article.get('link') or article.get('url') or "#"
            
            # Some versions might have a 'clickThroughUrl' dictionary
            if isinstance(link, dict):
                link = link.get('url', '#')
            
            # Basic sentiment analysis on the title
            score = analyzer.polarity_scores(title)['compound']
            sentiments.append(score)
            
            news_list.append({
                'title': title,
                'publisher': publisher,
                'link': link,
                'sentiment': score
            })

        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        print(f"Fetched {len(news_list)} news articles for {symbol}. Avg Sentiment: {avg_sentiment}")
        return avg_sentiment, news_list
    except Exception as e:
        print(f"Error fetching news for {symbol}: {e}")
        return 0, []

def get_market_movers():
    """
    Fetch top gainers and losers. 
    Using a sample of major tickers to find moves.
    """
    # Major S&P 100 or popular tickers for demo
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "BRK-B", "UNH", "JNJ", 
               "V", "XOM", "WMT", "JPM", "PG", "MA", "LLY", "AVGO", "HD", "CVX", 
               "PEP", "COST", "MRK", "KO", "ORCL", "BAC", "ADBE", "TMO", "CSCO", "CRM"]
    
    try:
        print("Fetching market movers...")
        # Download 2 days of data to calculate change
        data = yf.download(tickers, period="2d", group_by='ticker', progress=False)
        
        movers = []
        for ticker in tickers:
            try:
                if ticker in data and len(data[ticker]) >= 2:
                    current = data[ticker]['Close'].iloc[-1]
                    prev = data[ticker]['Close'].iloc[-2]
                    change = ((current - prev) / prev) * 100
                    movers.append({'symbol': ticker, 'price': current, 'change': change})
            except:
                continue
        
        # Sort by change
        movers.sort(key=lambda x: x['change'], reverse=True)
        top_gainers = movers[:10]
        top_losers = sorted(movers, key=lambda x: x['change'])[:10]
        
        return top_gainers, top_losers
    except Exception as e:
        print(f"Error fetching market movers: {e}")
        return [], []

if __name__ == "__main__":
    # Quick test
    test_symbol = "AAPL"
    print(f"Testing data fetcher for {test_symbol}...")
    
    hist = fetch_historical_data(test_symbol)
    if hist is not None:
        print(f"Fetched {len(hist)} rows of historical data.")
    
    avg_sent, news = fetch_news_sentiment(test_symbol)
    
    gainers, losers = get_market_movers()
    print(f"Top Gainer: {gainers[0] if gainers else 'None'}")
    print(f"Top Loser: {losers[0] if losers else 'None'}")
