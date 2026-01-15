import pandas as pd

class StockPredictor:
    def predict(self, historical_df, sentiment_score):
        if historical_df is None or historical_df.empty:
            return "NEUTRAL", "Insufficient data for a reliable forecast."

        df = historical_df.copy()
        # Explicit type conversion to avoid 'str' errors
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df = df.dropna(subset=['Close'])
        
        if df.empty:
            return "NEUTRAL", "Data processing resulted in an empty set."

        # SMA 5
        df['SMA_5'] = df['Close'].rolling(window=min(5, len(df))).mean()
        
        # Simple RSI calculation
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=min(14, len(df))).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=min(14, len(df))).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        latest_close = df['Close'].iloc[-1]
        latest_sma5 = df['SMA_5'].iloc[-1]
        latest_rsi = df['RSI'].iloc[-1] if not pd.isna(df['RSI'].iloc[-1]) else 50
        
        # Technical Signal
        if latest_close >= latest_sma5 and latest_rsi < 70:
            tech_signal = "BULLISH"
        elif latest_close < latest_sma5 or latest_rsi > 70:
            tech_signal = "BEARISH"
        else:
            tech_signal = "NEUTRAL"
            
        # Sentiment Signal
        sent_signal = "BULLISH" if sentiment_score > 0.1 else ("BEARISH" if sentiment_score < -0.1 else "NEUTRAL")
            
        # Decision Matrix
        if tech_signal == "BULLISH" and sent_signal == "BULLISH":
            return "UP", "Strong convergence of positive technical momentum and news sentiment."
        elif tech_signal == "BEARISH" and sent_signal == "BEARISH":
            return "DOWN", "Bearish technical indicators aligned with negative market sentiment."
        elif tech_signal == "BULLISH":
            return "UP", f"Technical indicators show strength ({tech_signal}) despite mixed sentiment."
        elif sent_signal == "BULLISH":
            return "UP", "Strong positive sentiment is currently driving the market outlook."
        elif tech_signal == "BEARISH":
            return "DOWN", "Technical indicators signal caution as momentum slows."
        else:
            return "NEUTRAL", "Market is currently showing sideways movement with balanced sentiment."

if __name__ == "__main__":
    # Mock test
    predictor = StockPredictor()
    mock_df = pd.DataFrame({'Close': [100, 102, 101, 103, 105]})
    sent = 0.5
    pred, reason = predictor.predict(mock_df, sent)
    print(f"Prediction: {pred}, Reason: {reason}")
