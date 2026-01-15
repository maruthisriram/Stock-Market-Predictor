from abc import ABC, abstractmethod
import yfinance as yf
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import numpy as np
import os
import requests
import xml.etree.ElementTree as ET
from html import unescape

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

try:
    from mftool import Mftool
except ImportError:
    Mftool = None

class BaseFetcher(ABC):
    """Abstract base class for all data fetchers"""
    def __init__(self):
        self._vader_analyzer = SentimentIntensityAnalyzer()
        self._finbert_pipe = None
        self._summarizer_pipe = None

    def get_finbert_pipeline(self):
        if pipeline is None: return None
        if self._finbert_pipe is None:
            try:
                print("Initializing FinBERT model...")
                self._finbert_pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert")
            except Exception as e:
                print(f"Error loading FinBERT: {e}")
                return None
        return self._finbert_pipe

    def get_summarizer_pipeline(self):
        if pipeline is None: return None
        if self._summarizer_pipe is None:
            try:
                print("Initializing Summarizer model...")
                self._summarizer_pipe = pipeline("summarization", model="t5-small")
            except Exception as e:
                print(f"Error loading Summarizer: {e}")
                return None
        return self._summarizer_pipe

    def _scrape_article_text(self, url):
        if Article is None: return None
        try:
            article = Article(url)
            article.download()
            article.parse()
            return article.text
        except Exception as e:
            return None

    def _fetch_google_news_rss(self, query):
        """Fallback to Google News RSS if yfinance fails or returns empty"""
        try:
            # hl=en-IN&gl=IN&ceid=IN:en for Global/India news
            url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
            response = requests.get(url, timeout=10)
            if response.status_code != 200: return []
            
            root = ET.fromstring(response.content)
            news_items = []
            for item in root.findall('.//item')[:8]:
                title = item.find('title').text if item.find('title') is not None else ""
                link = item.find('link').text if item.find('link') is not None else "#"
                source = item.find('source').text if item.find('source') is not None else "Google News"
                
                news_items.append({
                    'title': unescape(title),
                    'link': link,
                    'publisher': source,
                    'summary': title
                })
            return news_items
        except Exception as e:
            print(f"RSS fetch error: {e}")
            return []

    def fetch_news_sentiment(self, query, company_name=None):
        """Generic news fetching and sentiment analysis with Hybrid Scoring"""
        try:
            ticker = yf.Ticker(query)
            if not company_name:
                try:
                    info = ticker.info
                    company_name = info.get('longName') or info.get('shortName') or query
                except: company_name = query

            yf_news = getattr(ticker, 'news', [])
            news_to_process = []
            
            # Strict Relevance Check
            search_terms = {query.lower()}
            if company_name:
                clean_name = company_name.split(' - ')[0].split(' (')[0].split(',')[0].strip()
                search_terms.add(clean_name.lower())
                first_word = clean_name.split()[0]
                if len(first_word) > 3: search_terms.add(first_word.lower())

            for article in yf_news:
                title = (article.get('title') or "").lower()
                # Check title relevance
                if any(term in title for term in search_terms):
                    news_to_process.append(article)

            # RSS Fallback if sparse
            if len(news_to_process) < 3:
                # Broader query for RSS (esp. for MFs)
                words = company_name.split()
                rss_query = " ".join(words[:3]) if len(words) > 3 else company_name
                
                rss_hits = self._fetch_google_news_rss(rss_query)
                for item in rss_hits:
                    # Apply slightly smarter relevance to RSS hits
                    item_title = item['title'].lower()
                    # Hit is relevant if it matches ticker OR any significant word from company name
                    is_match = any(term in item_title for term in search_terms)
                    if is_match:
                        news_to_process.append({
                            'title': item['title'],
                            'link': item['link'],
                            'publisher': item['publisher'],
                            'summary': item['title']
                        })

            if not news_to_process: return 0, []

            nlp = self.get_finbert_pipeline()
            summarizer = self.get_summarizer_pipeline()
            sentiments = []
            news_list = []
            
            for article in news_to_process[:5]:
                title = article.get('title') or "Financial Report"
                preview = article.get('summary') or article.get('description') or ""
                publisher = article.get('publisher') or "Market Media"
                link = article.get('link') or article.get('url') or "#"
                if isinstance(link, dict): link = link.get('url', '#')
                
                generated_summary = preview if preview else title
                full_text = None
                
                # Attempt Scrape for content
                if link != "#" and summarizer and "google.com" not in link:
                    full_text = self._scrape_article_text(link)
                    if full_text and len(full_text) > 200:
                        try:
                            summary_input = f"summarize: {full_text[:1000]}"
                            gen_sum = summarizer(summary_input, max_length=120, min_length=40, do_sample=False)
                            generated_summary = gen_sum[0]['summary_text']
                        except: pass

                # Hybrid Sentiment Analysis
                score = 0
                reasoning = "Neutral outlook."
                sentiment_input = (full_text[:1500] if full_text else f"{title}. {preview}")[:1500]
                
                vader_score = self._vader_analyzer.polarity_scores(title)['compound']
                
                if nlp:
                    try:
                        res = nlp(sentiment_input[:512])[0]
                        label = res['label'].lower()
                        finbert_score = res['score'] if label == 'positive' else (-res['score'] if label == 'negative' else 0)
                        
                        # Hybrid logic: If FinBERT is neutral, use VADER to get more granularity
                        if label == 'neutral' or abs(finbert_score) < 0.1:
                            score = vader_score # Fallback/Hybrid
                            reasoning = "Neutral alert." if abs(score) < 0.05 else ("Positive signals." if score > 0 else "Negative signals.")
                        else:
                            score = finbert_score
                            reasoning = f"{label.capitalize()} outlook."
                    except:
                        score = vader_score
                else:
                    score = vader_score
                
                sentiments.append(score)
                news_list.append({
                    'title': title,
                    'summary': generated_summary,
                    'publisher': publisher,
                    'link': link,
                    'sentiment': score,
                    'reasoning': reasoning
                })

            avg_score = sum(sentiments) / len(sentiments) if sentiments else 0
            return avg_score, news_list
        except Exception as e:
            print(f"News fetch error: {e}")
            return 0, []

    @abstractmethod
    def fetch_historical_data(self, identifier, period="1mo", interval="1d"):
        pass

    @abstractmethod
    def search(self, query):
        pass

class StockFetcher(BaseFetcher):
    def fetch_historical_data(self, symbol, period="1mo", interval="1d"):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            return df if not df.empty else None
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None

    def search(self, query):
        try:
            if not query or len(query) < 2: return []
            search_obj = yf.Search(query)
            results = []
            for quote in getattr(search_obj, 'quotes', []):
                sym = quote.get('symbol')
                name = quote.get('shortname') or quote.get('longname') or sym
                if sym: results.append({'symbol': sym, 'name': name})
            return results[:10]
        except Exception as e:
            print(f"Error searching for {query}: {e}")
            return []

    def get_market_movers(self):
        return [
            {'symbol': 'RELIANCE.NS', 'change': 1.2},
            {'symbol': 'HDFCBANK.NS', 'change': 0.8},
            {'symbol': 'TCS.NS', 'change': -0.5},
            {'symbol': 'AAPL', 'change': 2.1},
            {'symbol': 'TSLA', 'change': -1.4}
        ], []

class MutualFundFetcher(BaseFetcher):
    _mf_instance = None
    def __init__(self):
        super().__init__()
        if Mftool and MutualFundFetcher._mf_instance is None:
            print("Initializing Mftool (Mutual Fund Data)...")
            MutualFundFetcher._mf_instance = Mftool()
        self.mf = MutualFundFetcher._mf_instance

    def fetch_historical_data(self, identifier, period="1mo", interval="1d"):
        if len(identifier) == 5 and identifier.endswith('X'):
            try:
                ticker = yf.Ticker(identifier)
                df = ticker.history(period=period, interval=interval)
                return df if not df.empty else None
            except: return None
        if self.mf and identifier.isdigit():
            try:
                data = self.mf.get_scheme_historical_nav(identifier, as_Dataframe=True)
                if data is None or data.empty: return None
                df = data.rename(columns={'nav': 'Close'})
                df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
                df = df.dropna(subset=['Close'])
                df.index = pd.to_datetime(df.index, dayfirst=True)
                df = df.sort_index()
                days = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365}.get(period, 30)
                df = df[df.index > (df.index.max() - timedelta(days=days))]
                return df
            except Exception as e:
                print(f"Error fetching Indian MF data: {e}")
                return None
        return None

    def get_trending_mfs(self):
        """Fetch real trending funds by calculating recent performance"""
        codes = {
            "120716": "HDFC Index Nifty 50",
            "119842": "SBI Bluechip Fund",
            "118834": "Nippon India Large Cap",
            "100355": "ICICI Prudential Bluechip",
            "VFIAX": "Vanguard 500 Index",
            "FXAIX": "Fidelity 500 Index"
        }
        
        results = []
        for code, name in codes.items():
            try:
                # Use fetch_historical_data which already handles the logic
                # We use 1mo period to get enough points for change calculation
                df = self.fetch_historical_data(code, period="1mo")
                if df is not None and not df.empty and len(df) >= 2:
                    # Sort just in case
                    df = df.sort_index()
                    latest = df['Close'].iloc[-1]
                    prev = df['Close'].iloc[-2]
                    change = ((latest - prev) / prev) * 100
                    results.append({'name': name, 'symbol': code, 'change': round(change, 2)})
            except: continue
        
        # Sort by best performers for "Trending" feel
        return sorted(results, key=lambda x: x['change'], reverse=True)[:5]

    def search(self, query):
        if not self.mf: return []
        try:
            all_schemes = self.mf.get_scheme_codes()
            results = []
            query_lower = query.lower()
            count = 0
            for code, name in all_schemes.items():
                if query_lower in name.lower():
                    results.append({'symbol': code, 'name': name})
                    count += 1
                    if count >= 8: break
            us_search = yf.Search(query + " Mutual Fund")
            for quote in getattr(us_search, 'quotes', []):
                symbol = quote.get('symbol')
                if symbol and symbol.endswith('X'):
                    results.append({'symbol': symbol, 'name': quote.get('shortname', symbol)})
                    if len(results) >= 12: break
            return results
        except: return []

class DataFetcher:
    def __init__(self):
        self._shared_fetcher = StockFetcher()
    def get_finbert_pipeline(self): return self._shared_fetcher.get_finbert_pipeline()
    def get_summarizer_pipeline(self): return self._shared_fetcher.get_summarizer_pipeline()
    def _scrape_article_text(self, url): return self._shared_fetcher._scrape_article_text(url)
    def _vader_analyzer(self): return self._shared_fetcher._vader_analyzer
    def get_market_movers(self): return self._shared_fetcher.get_market_movers()
