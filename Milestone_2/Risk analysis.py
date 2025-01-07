import pandas as pd
import requests
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import json
class DataSource:
    """Handles data collection from various sources"""
    def __init__(self, apiKeys):
        self.apiKeys = apiKeys
        self.setupLogging()

    def setupLogging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("supply_chain_monitor.log"),
                logging.StreamHandler()
            ]
        )

    def collect_news(self, query, days_back=7):
        """Collect news using NewsAPI"""
        logging.info(f"Collecting news for query: {query}")
        NEWS_API_KEY = self.apiKeys.get('news_api')
        if not NEWS_API_KEY:
            raise ValueError("NewsAPI key is missing!")

        # Date range for fetching news
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)

        # Parameters for the API request
        params = {
            'q': query,
            'from': start_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'to': end_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'apiKey': NEWS_API_KEY,
            'language': 'en',
            'sortBy': 'relevancy',
            'pageSize': 100  # Fetch up to 100 articles per request
        }

        try:
            response = requests.get("https://newsapi.org/v2/everything", params=params)
            response.raise_for_status()
            articles = response.json().get('articles', [])

            if not articles:
                logging.warning("No articles found for the given query and date range.")
                return pd.DataFrame()

            # Process the articles into a DataFrame
            data = [{
                'title': article.get('title'),
                'content': article.get('content'),
                'description': article.get('description'),
                'publishedAt': article.get('publishedAt'),
                'source': article.get('source', {}).get('name'),
                'url': article.get('url')
            } for article in articles]

            return pd.DataFrame(data)

        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching news articles: {e}")
            return pd.DataFrame()


class RiskAnalyzer:
    """Analyzes supply chain risks using Hugging Face models"""
    def __init__(self):
        self.model_name = "finiteautomata/bertweet-base-sentiment-analysis"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

    def analyze_sentiment(self, text):
        """Analyze sentiment of the given text"""
        logging.info("Analyzing sentiment...")
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        outputs = self.model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        sentiment = torch.argmax(predictions, dim=-1).item()

        # 0: Negative, 1: Neutral, 2: Positive (specific to this model)
        if sentiment == 0:
            return "Negative"
        elif sentiment == 1:
            return "Neutral"
        else:
            return "Positive"

    def extract_trends(self, texts, n_topics=5):
        """Extract emerging trends using topic modeling"""
        logging.info("Extracting trends from collected data...")

        # Use TF-IDF to vectorize the text
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)

        # Apply NMF for topic modeling
        nmf_model = NMF(n_components=n_topics, random_state=42)
        nmf_matrix = nmf_model.fit_transform(tfidf_matrix)

        # Get the top words for each topic
        feature_names = vectorizer.get_feature_names_out()
        trends = {}
        for topic_idx, topic in enumerate(nmf_model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[-10:]]
            trends[f"Topic {topic_idx+1}"] = top_words

        logging.info(f"Extracted Trends: {trends}")
        return trends


class SupplyChainMonitor:
    """Main class for supply chain monitoring system"""
    def __init__(self, config):
        self.config = config
        self.dataSource = DataSource(config.get('apiKeys', {}))
        self.riskAnalyzer = RiskAnalyzer()

    def monitor_market(self, product):
        """Monitor the market for a specific product"""
        logging.info(f"Monitoring market for product: {product}")

        # Step 1: Collect News
        news_df = self.dataSource.collect_news(query=product, days_back=7)

        if news_df.empty:
            logging.info("No data collected, skipping analysis.")
            return
        
        # Step 2: Analyze Sentiment
        news_df['Sentiment'] = news_df['content'].fillna('').apply(self.riskAnalyzer.analyze_sentiment)

        # Step 3: Summarize Results
        sentiment_summary = news_df['Sentiment'].value_counts()
        logging.info(f"Sentiment Summary for {product}:{sentiment_summary}")

        overall_sentiment = "Positive" if sentiment_summary.get("Positive", 0) > sentiment_summary.get("Negative", 0) else "Negative"
        logging.info(f"Overall Market Sentiment for {product}: {overall_sentiment}")

        # Step 4: Extract Emerging Trends
        trends = self.riskAnalyzer.extract_trends(news_df['content'].dropna().tolist())

        # Save detailed results
        news_df.to_csv(f"{product}_market_sentiment.csv", index=False)
        logging.info(f"Detailed sentiment analysis saved to {product}_market_sentiment.csv")
        with open(f"{product}_trends.json", "w") as f:
            json.dump(trends, f, indent=4)
        logging.info(f"Extracted trends saved to {product}_trends.json")


def main():
    config = {'apiKeys': {'news_api': '909999c59129445ab2c3904a9d7786b5'}}  # Replace with your actual API key
    monitor = SupplyChainMonitor(config)

    # Monitor the GPU market
    monitor.monitor_market("GPU")


if __name__ == "__main__":
    main()
