import yfinance as yf
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import numpy as np
try:
    from transformers import pipeline
except ImportError:
    pipeline = None

try:
    from newspaper import Article
    import nltk
    nltk.download('punkt', quiet=True)
except ImportError:
    Article = None

# Global model initialization for sentiment
FINBERT_PIPE = None
SUMMARIZER_PIPE = None

def get_finbert_pipeline():
    global FINBERT_PIPE
    if pipeline is None:
        return None
    if FINBERT_PIPE is None:
        try:
            print("Initializing FinBERT model (Meaning-aware Analysis)...")
            FINBERT_PIPE = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        except Exception as e:
            print(f"Error loading FinBERT: {e}")
            return None
    return FINBERT_PIPE

def get_summarizer_pipeline():
    global SUMMARIZER_PIPE
    if pipeline is None:
        return None
    if SUMMARIZER_PIPE is None:
        try:
            print("Initializing Summarizer model (Article Summarization)...")
            # Using t5-small for speed and reasonable summaries
            SUMMARIZER_PIPE = pipeline("summarization", model="t5-small")
        except Exception as e:
            print(f"Error loading Summarizer: {e}")
            return None
    return SUMMARIZER_PIPE

def scrape_article_text(url):
    """
    Fetch and extract full text from an article URL using newspaper3k.
    """
    if Article is None:
        return None
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

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

def search_stocks(query):
    """
    Search for stocks using a string query (name or ticker).
    Returns a list of dicts with 'symbol' and 'name'.
    """
    try:
        if not query or len(query) < 2:
            return []
        search = yf.Search(query)
        results = []
        for quote in getattr(search, 'quotes', []):
            symbol = quote.get('symbol')
            name = quote.get('shortname') or quote.get('longname') or symbol
            if symbol:
                results.append({'symbol': symbol, 'name': name})
        return results[:10]  # Limit to 10 results
    except Exception as e:
        print(f"Error searching for {query}: {e}")
        return []

def fetch_news_sentiment(symbol):
    """
    Fetch recent news and calculate average sentiment score using full-article context and generate true summaries.
    """
    try:
        print(f"Fetching news for {symbol}...")
        ticker = yf.Ticker(symbol)
        
        # Get company name for better filtering
        company_name = symbol
        try:
            info = ticker.info
            company_name = info.get('longName') or info.get('shortName') or symbol
        except:
            pass
            
        news = getattr(ticker, 'news', [])
        
        if not news:
            print(f"No news found for {symbol}.")
            return 0, []

        print(f"Company: {company_name} ({symbol})")
        
        # Filter news for relevance
        relevant_news = []
        for article in news:
            content = article.get('content', {})
            data_source = content if content else article
            title = (data_source.get('title') or article.get('title') or "").lower()
            preview = (data_source.get('summary') or data_source.get('description') or "").lower()
            
            # Check if symbol or company name (or parts of it) are mentioned
            search_terms = {symbol.lower()}
            if company_name:
                # Add the first few words of the company name to search terms (e.g., "Apple" from "Apple Inc.")
                name_words = company_name.split()
                if name_words:
                    search_terms.add(name_words[0].lower())
                search_terms.add(company_name.lower())
            
            is_relevant = any(term in title or term in preview for term in search_terms)
            
            if is_relevant:
                relevant_news.append(article)
        
        print(f"Filtered {len(news)} total articles down to {len(relevant_news)} relevant articles.")
        
        if not relevant_news:
            # If no relevant news found, maybe the filter was too strict, 
            # but user specifically asked for relevance so we'll stick to it.
            return 0, []

        news = relevant_news

        # Try to get models
        nlp = get_finbert_pipeline()
        summarizer = get_summarizer_pipeline()
        vader_analyzer = SentimentIntensityAnalyzer() if nlp is None else None
        
        sentiments = []
        news_list = []

        # Limit deep analysis to top 5 articles for performance
        analysis_limit = 5
        
        for i, article in enumerate(news):
            content = article.get('content', {})
            data_source = content if content else article

            title = data_source.get('title') or article.get('title') or ""
            preview = data_source.get('summary') or data_source.get('description') or ""
            
            if not title:
                continue
            
            publisher = data_source.get('publisher') or article.get('publisher') or "Market News"
            link = data_source.get('link') or data_source.get('url') or "#"
            if isinstance(link, dict):
                link = link.get('url', '#')
            
            full_text = None
            generated_summary = preview # Fallback to preview
            
            # Deep analysis for top articles
            if i < analysis_limit and link != "#":
                print(f"Deep analyzing article {i+1}: {title[:50]}...")
                full_text = scrape_article_text(link)
                
                if full_text and len(full_text) > 100:
                    # Generate a true summary if we have enough text
                    if summarizer:
                        try:
                            # T5 needs 'summarize: ' prefix
                            summary_input = f"summarize: {full_text[:1000]}"
                            gen_sum = summarizer(summary_input, max_length=100, min_length=30, do_sample=False)
                            generated_summary = gen_sum[0]['summary_text']
                        except Exception as se:
                            print(f"Summarization failed: {se}")
            
            # Use full text for sentiment if available, else combine title and preview
            sentiment_input = full_text[:1500] if full_text else f"{title}. {preview}"
            
            score = 0
            reasoning = "Neutral sentiment indicates low immediate impact from this news."
            
            if nlp:
                try:
                    # Truncate for BERT limit
                    result = nlp(sentiment_input[:512])[0]
                    label = result['label'].lower()
                    confidence = result['score']
                    
                    if label == 'positive':
                        score = confidence
                        reasoning = "The full article indicates a positive trend or strong growth potential."
                    elif label == 'negative':
                        score = -confidence
                        reasoning = "The article content highlights significant risks or a bearish outlook."
                except:
                    score = SentimentIntensityAnalyzer().polarity_scores(title)['compound']
            else:
                score = vader_analyzer.polarity_scores(title)['compound']
                
            sentiments.append(score)
            news_list.append({
                'title': title,
                'summary': generated_summary, # Genuine summary
                'publisher': publisher,
                'link': link,
                'sentiment': score,
                'reasoning': reasoning
            })

        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        print(f"Analyzed {len(news_list)} news articles for {symbol}. Contextual Sentiment: {avg_sentiment:.2f}")
        return avg_sentiment, news_list
    except Exception as e:
        print(f"Error fetching news for {symbol}: {e}")
        return 0, []

def get_market_movers():
    """
    Fetch top gainers and losers. 
    Using a sample of major tickers to find moves.
    """
    # Mix of US (S&P 100) and Indian (Nifty 50) popular tickers
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", 
               "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
               "BRK-B", "UNH", "JNJ", "V", "XOM", "WMT", "JPM", "PG", "MA", "LLY"]
    
    try:
        print("Fetching market movers...")
        # Download a bit more data to ensure we have at least 2 valid points
        data = yf.download(tickers, period="5d", group_by='ticker', progress=False)
        
        movers = []
        for ticker in tickers:
            try:
                if ticker in data:
                    # Drop NaNs and get the last two valid close prices
                    prices = data[ticker]['Close'].dropna()
                    if len(prices) >= 2:
                        current = prices.iloc[-1]
                        prev = prices.iloc[-2]
                        change = ((current - prev) / prev) * 100
                        
                        # Only add if change is not nan or inf
                        import numpy as np
                        if not np.isnan(change) and not np.isinf(change):
                            movers.append({'symbol': ticker, 'price': current, 'change': change})
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                continue
        
        # Sort and return
        movers.sort(key=lambda x: x['change'], reverse=True)
        top_gainers = movers[:10]
        top_losers = sorted(movers, key=lambda x: x['change'])[:10]
        
        return top_gainers, top_losers
    except Exception as e:
        print(f"Error fetching market movers: {e}")
        return [], []

def get_sector_performance():
    """
    Estimate sector performance using major ETFs or large-cap averages.
    """
    sectors = {
        "Technology": "XLK",
        "Financials": "XLF",
        "Healthcare": "XLV",
        "Energy": "XLE",
        "Consumer Disc": "XLY",
        "Consumer Staples": "XLP",
        "Utilities": "XLU",
        "Real Estate": "XLRE",
        "Materials": "XLB",
        "Communication": "XLC"
    }
    
    try:
        data = yf.download(list(sectors.values()), period="2d", progress=False)['Close']
        performance = []
        for name, ticker in sectors.items():
            if ticker in data and len(data[ticker]) >= 2:
                current = data[ticker].iloc[-1]
                prev = data[ticker].iloc[-2]
                change = ((current - prev) / prev) * 100
                performance.append({'sector': name, 'change': change})
        
        performance.sort(key=lambda x: x['change'], reverse=True)
        return performance
    except Exception as e:
        print(f"Error fetching sector performance: {e}")
        return []

if __name__ == "__main__":
    # Quick test
    test_symbol = "AAPL"
    print(f"Testing data fetcher for {test_symbol}...")
    
    hist = fetch_historical_data(test_symbol)
    if hist is not None:
        print(f"Fetched {len(hist)} rows of historical data.")
    
    avg_sent, news = fetch_news_sentiment(test_symbol)
    if news:
        print(f"Sample news sentiment analysis for {test_symbol}:")
        for i, item in enumerate(news[:3]):
            print(f"{i+1}. {item['title']} -> Sentiment Score: {item['sentiment']:.2f}")
    
    gainers, losers = get_market_movers()
    print(f"Top Gainer: {gainers[0]['symbol'] if gainers else 'None'}")
    print(f"Top Loser: {losers[0]['symbol'] if losers else 'None'}")
