import logging
import time

import streamlit as st

from app.sentiment_analyzer import SentimentAnalyzer
from app.utility import new_line, parse_info, parse_comments_dataset, plot_comments_replies_trend, SAMPLE_URL
from app.youtube_data import YouTubeData

# Init logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

sentiment = SentimentAnalyzer()

# Configure the page
st.set_page_config(page_title="YouTube Sentiment Analyzer", page_icon=None, layout="centered")

# App title
st.title(":red[YouTube Sentiment Analyzer]")
new_line()

st.info("The processing time for analysis depends on the size of comments. The more comments there are, "
        "the longer it may take.")
new_line()

with st.form(key="input_form"):
    new_line()
    yt_url = st.text_input("Enter YouTube URL", value=SAMPLE_URL)
    st.caption("Sample URL: https://www.youtube.com/watch?v=X3paOmcrTjQ")
    new_line()
    submit_btn = st.form_submit_button("Analyze", use_container_width=True)
    new_line()

if submit_btn and yt_url:
    with st.spinner("Analyzing... Please wait."):
        try:
            # StartTime: Calculate process time
            start_time = time.time()

            # Collect YouTube video details
            dataset = YouTubeData(yt_url)
            info = dataset.get_video_info()
            # Comments Dataframe
            comments_df = dataset.get_dataframes()[0]

            # Replies Dataframe
            replies_df = dataset.get_dataframes()[1]

            # EndTime: Calculate process time
            end_time = time.time()
            new_line()
            st.success(f"Analysis finished in {round(end_time - start_time, 2)}s")
            new_line(3)

            # 1. Parse and display video info
            parse_info(info, len(comments_df.index))

            # 2. Plot comments & replies trends in a chart
            plot_comments_replies_trend(comments_df, replies_df)

            # 3. Parse and display the sample dataset
            parse_comments_dataset(comments_df)

            # 4. Analyze the sentiment
            sentiment.set_data(comments_df, replies_df)
            sentiment.analyze_sentiment()

            # 5. Display sentiment report
            sentiment.show_report_and_plot()

        except Exception as ex:
            new_line()
            error_msg = str(ex)
            st.error(error_msg)
