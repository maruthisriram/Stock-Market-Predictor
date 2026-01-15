# StockPredictorApp System Design

This document describes the Object-Oriented Design of the StockPredictorApp.

```mermaid
classDiagram
    class DataFetcher {
        -finbert_pipe: pipeline
        -summarizer_pipe: pipeline
        +__init__()
        +get_finbert_pipeline()
        +get_summarizer_pipeline()
        +fetch_historical_data(symbol, period, interval)
        +fetch_news_sentiment(symbol)
        +search_stocks(query)
        +get_market_movers()
        +get_sector_performance()
        -_scrape_article_text(url)
    }

    class StockPredictor {
        +predict(historical_df, sentiment_score)
    }

    class DashboardApp {
        -fetcher: DataFetcher
        -predictor: StockPredictor
        +__init__()
        +apply_styles()
        +set_symbol(new_sym)
        +render_sidebar()
        +render_metrics(df, avg_sentiment)
        +render_prediction_card(prediction, reason)
        +render_tabs(df, news_articles)
        +render_market_overview()
        +run()
    }

    DashboardApp --> DataFetcher : uses
    DashboardApp --> StockPredictor : uses
```

## Component Descriptions

- **DataFetcher**: Responsible for interacting with external data sources (Yahoo Finance, news articles) and running AI logic for sentiment analysis and summarization.
- **StockPredictor**: Contains the business logic for predicting stock movement based on technical indicators and sentiment scores.
- **DashboardApp**: The main UI component built with Streamlit, coordinating user input, data fetching, and prediction display.
