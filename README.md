# Emo Gauge
A real-time sentiment analysis tool for YouTube video comments that leverages deep learning to classify viewer sentiment and visualize engagement patterns.

## Overview

Emo Gauge analyzes YouTube video comments using an LSTM-based deep learning model to classify sentiment as Positive, Negative, or Neutral. The application retrieves comments via the YouTube Data API, processes them through a trained neural network, and presents interactive visualizations showing sentiment distribution, trends over time, and engagement metrics.

## Features

- **Deep Learning Sentiment Classification** – LSTM model trained on labeled comment data for accurate sentiment prediction
- **YouTube Data Integration** – Fetches video details, comments, and replies using YouTube Data API v3
- **Real-Time Analysis** – Processes comments and generates sentiment insights in real-time
- **Interactive Visualizations** – Pie charts, bar charts, and time-series plots for sentiment distribution and trends
- **Comment Preprocessing** – Cleans data by removing URLs, mentions, special characters, and stopwords
- **Engagement Metrics** – Displays video views, likes, and comment statistics
- **Time-Series Analysis** – Tracks sentiment changes over time with monthly aggregation
- **User-Friendly Interface** – Built with Streamlit for intuitive interaction and visualization

## Tech Stack

- **Backend:** Python, Streamlit
- **Machine Learning:** Keras, TensorFlow, LSTM, Tokenization
- **Data Processing:** Pandas, NumPy, NLTK, Scikit-learn
- **Visualization:** Plotly, Altair
- **API Integration:** Google YouTube Data API v3

## Installation
```bash
# Clone the repository
git clone https://github.com/DevelopedBy-Siva/emo-gauge.git
cd emo-gauge

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export GOOGLE_API_KEY="your_youtube_api_key"

# Run the application
streamlit run main.py
```

## Project Structure
```
emo-gauge/
├── app/
│   ├── sentiment_analyzer.py    # Core sentiment analysis logic
│   ├── youtube_data.py          # YouTube API integration
│   └── utility.py               # Helper functions and visualization
├── train/
│   ├── train_model.py           # LSTM model training script
│   ├── yt_model.h5             # Trained model weights
│   └── tokenizer.pkl           # Saved tokenizer
├── data/
│   └── dataset.csv             # Training dataset
└── main.py                     # Streamlit application entry point
```

## How It Works

1. **Data Retrieval** – Fetches video metadata, comments, and replies using YouTube Data API
2. **Text Preprocessing** – Cleans comments by removing URLs, mentions, punctuation, and stopwords
3. **Tokenization** – Converts text into sequences using a pre-trained tokenizer
4. **Sentiment Prediction** – LSTM model classifies each comment as Positive, Neutral, or Negative
5. **Visualization** – Generates interactive charts showing sentiment distribution and temporal trends
6. **Analysis Report** – Displays comprehensive sentiment breakdown with metrics and insights

## Model Architecture

- **Embedding Layer** – 5000 vocabulary size, 128-dimensional embeddings
- **Spatial Dropout** – 0.2 dropout rate for regularization
- **LSTM Layer** – 100 units with 0.2 dropout and recurrent dropout
- **Dense Output Layer** – 3 units with softmax activation (Positive, Neutral, Negative)
- **Training** – 20 epochs, batch size 64, categorical crossentropy loss

## API Requirements

Requires a Google YouTube Data API v3 key. Set as environment variable:
```bash
export GOOGLE_API_KEY="your_api_key_here"
```

## Usage

1. Launch the Streamlit app
2. Enter a YouTube video URL
3. Click "Analyze" to process comments
4. View sentiment distribution, engagement metrics, and trend visualizations
5. Explore interactive charts with filtering options

## Features in Detail

- **Sentiment Pie Chart** – Percentage breakdown of positive, negative, and neutral comments
- **Sentiment Breakdown Table** – Individual comment classifications with original text
- **Distribution Bar Chart** – Count of comments by sentiment category
- **Sentiment Over Time** – Monthly trend lines showing sentiment evolution
- **Engagement Trends** – Comments and replies activity over time
- **Video Metrics** – Views, likes, and comment count with formatted numbers

## Future Enhancements

- Multi-language sentiment analysis support
- Toxicity and emotion detection
- Comparative analysis across multiple videos
- Export functionality for analysis reports
- Real-time streaming analysis for live videos

