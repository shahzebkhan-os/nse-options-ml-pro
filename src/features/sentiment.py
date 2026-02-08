import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from src.utils.logger import get_logger

logger = get_logger(__name__)

class SentimentEngine:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def get_sentiment(self, symbol):
        """
        Fetches news from Yahoo Finance and calculates average sentiment.
        Returns: Score (-1 to 1) and a Label (Bullish/Bearish/Neutral).
        """
        ticker_name = f"{symbol}.NS" if not symbol.endswith(".NS") else symbol
        try:
            ticker = yf.Ticker(ticker_name)
            news = ticker.news
            
            if not news:
                return 0, "NEUTRAL", []

            scores = []
            headlines = []
            
            for article in news[:5]: # Analyze top 5 articles
                title = article.get('title', '')
                if title:
                    sentiment = self.analyzer.polarity_scores(title)
                    scores.append(sentiment['compound'])
                    headlines.append(title)

            if not scores:
                return 0, "NEUTRAL", []

            avg_score = sum(scores) / len(scores)
            
            if avg_score > 0.05:
                label = "BULLISH"
            elif avg_score < -0.05:
                label = "BEARISH"
            else:
                label = "NEUTRAL"
                
            return avg_score, label, headlines
            
        except Exception as e:
            logger.error(f"Sentiment error for {symbol}: {e}")
            return 0, "NEUTRAL", []
