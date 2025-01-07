# Milestone 2: Global Data Monitoring and Analysis Engine

This project aims to monitor the global supply chain market using Natural Language Processing (NLP) and machine learning techniques. Specifically, it collects and analyzes news data to identify emerging risks and trends in the supply chain landscape.

---

## Features

1. **Data Collection**:
   - Utilizes NewsAPI to fetch news articles based on specific queries (e.g., GPU market).
   - Filters articles by relevance, language, and date.

2. **Sentiment Analysis**:
   - Leverages the `bertweet-base-sentiment-analysis` model from Hugging Face for sentiment classification.
   - Classifies articles as Positive, Neutral, or Negative based on their content.

3. **Trend Identification**:
   - Applies Topic Modeling (TF-IDF and NMF) to extract and summarize emerging trends in the collected news articles.

4. **Logging and Persistence**:
   - Logs detailed information about the analysis process for transparency and debugging.
   - Saves sentiment results in a CSV file and trends in a JSON file.

---

## File Structure

```
|-- gpu_market_risk.py  # Main script containing the implementation
|-- requirements.txt    # List of dependencies for the project
|-- logs/               # Log files for monitoring execution (generated during runtime)
|-- outputs/            # Directory to save sentiment analysis and trends (generated during runtime)
```

---

## Requirements

- Python 3.8+

### Install Dependencies

Install the required Python packages using the following command:
```bash
pip install -r requirements.txt
```

### requirements.txt
```
pandas
requests
transformers
torch
scikit-learn
```

---

## Usage

1. **Set Up API Key**:
   - Obtain an API key from [NewsAPI](https://newsapi.org/).
   - Replace the placeholder in the `config` dictionary in `gpu_market_risk.py` with your API key.

2. **Run the Script**:
   Execute the main script to monitor the GPU market or any other specified product:
   ```bash
   python gpu_market_risk.py
   ```

3. **Outputs**:
   - Sentiment analysis results are saved in a CSV file (e.g., `GPU_market_sentiment.csv`).
   - Extracted trends are saved in a JSON file (e.g., `GPU_trends.json`).

---

## Example

### Sentiment Analysis Output
A CSV file summarizing the sentiment of collected articles:
```
| title                | content            | description        | sentiment |
|----------------------|--------------------|--------------------|-----------|
| Article Title 1      | ...                | ...                | Positive  |
| Article Title 2      | ...                | ...                | Negative  |
```

### Trend Identification Output
A JSON file with extracted trends:
```json
{
    "Topic 1": ["gpu", "shortage", "supply", "chain", "nvidia"],
    "Topic 2": ["demand", "market", "price", "production", "risk"]
}
```

---

## Logging
- Logs are stored in `supply_chain_monitor.log` and provide a detailed execution trace, including errors and process summaries.

---

## Customization
- To monitor a different product, change the argument in `monitor.monitor_market("GPU")` to your desired query.
- Modify parameters like `days_back` to adjust the date range for data collection.

---

## Contact
For questions or feedback, please contact [Your Name/Team Name].

