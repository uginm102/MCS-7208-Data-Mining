from mrjob.job import MRJob
import csv
import io
import logging
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download('vader_lexicon')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# initialize NLTK sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

log = logging.getLogger(__name__)


def get_sentiment(text):
    """
    Compute sentiment of raw text using VADER.
    Returns 'Positive', 'Neutral', or 'Negative' based on compound score.
    """
    scores = analyzer.polarity_scores(text)
    if scores['compound'] >= 0.5:
        return "Positive"
    elif scores['compound'] > -0.5:
        return "Neutral"
    else:
        return "Negative"

class AmazonReviews(MRJob):
    def mapper(self, _, line):
        """
        Parse CSV line, extract review text, and yield counts for total lines and sentiment.
        Assumes CSV format with 'review_body' in the 3rd column (index 2).
        """
        try:
            # Parse CSV line
            reader = csv.reader(io.StringIO(line))
            fields = next(reader)
            # Extract review text (adjust index based on actual dataset)
            review_text = fields[2]  # Example: assuming review_body is 3rd field
            # Count total lines
            yield "lines", 1
            # Count sentiment
            sentiment = get_sentiment(review_text)
            yield sentiment, 1
        except (IndexError, csv.Error) as e:
            # Log parsing errors and skip malformed lines
            log.warning(f"Skipping malformed line: {line} - Error: {e}")

    def reducer(self, key, values):
        """
        Sum the values for each key and yield the total.
        Avoids iterator exhaustion by summing directly.
        """
        total = sum(values)  # Sum consumes the iterator
        log.info(f"Received {total} reviews for key {key}")
        yield key, total


if __name__ == '__main__':
    AmazonReviews.run()