import pandas as pd

def predict_movement(historical_df, sentiment_score):
    """
    A simple prediction logic combining price trends and news sentiment.
    Returns: "UP", "DOWN", or "NEUTRAL" and a confidence/reasoning message.
    """
    if historical_df is None or historical_df.empty:
        return "NEUTRAL", "No historical data available."

    # Calculate 5-day and 20-day Simple Moving Averages (or whatever is available)
    historical_df['SMA_5'] = historical_df['Close'].rolling(window=min(5, len(historical_df))).mean()
    
    # Simple trend logic: is the latest price above the 5-day SMA?
    latest_close = historical_df['Close'].iloc[-1]
    latest_sma5 = historical_df['SMA_5'].iloc[-1]
    
    price_trend = "UP" if latest_close >= latest_sma5 else "DOWN"
    
    # Sentiment levels
    if sentiment_score > 0.1:
        sentiment_trend = "UP"
    elif sentiment_score < -0.1:
        sentiment_trend = "DOWN"
    else:
        sentiment_trend = "NEUTRAL"
        
    # Combine logic
    if price_trend == "UP" and sentiment_trend == "UP":
        prediction = "UP"
        reason = "Bullish trend and positive news sentiment."
    elif price_trend == "DOWN" and sentiment_trend == "DOWN":
        prediction = "DOWN"
        reason = "Bearish trend and negative news sentiment."
    elif sentiment_trend == "UP":
        prediction = "UP"
        reason = "Positive news sentiment outweighing technical trend."
    elif sentiment_trend == "DOWN":
        prediction = "DOWN"
        reason = "Negative news sentiment outweighing technical trend."
    else:
        prediction = price_trend
        reason = f"Following technical trend ({price_trend}) as sentiment is neutral."

    return prediction, reason

if __name__ == "__main__":
    # Mock test
    mock_df = pd.DataFrame({'Close': [100, 102, 101, 103, 105]})
    sent = 0.5
    pred, reason = predict_movement(mock_df, sent)
    print(f"Prediction: {pred}, Reason: {reason}")
