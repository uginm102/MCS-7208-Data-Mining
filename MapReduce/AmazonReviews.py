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


def get_sentiment(line):
    tokens = word_tokenize(line.lower())

    # Remove stop words
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Join the tokens back into a string
    processed_text = ' '.join(lemmatized_tokens)

    scores = analyzer.polarity_scores(processed_text)

    # Classify the text as positive, neutral, or negative
    if scores['compound'] >= 0.5:
        return "Positive"
    elif scores['compound'] > -0.5:
        return "Neutral"
    else:
        return "Negative"

def string_to_csv_line(s):
    # Create a string buffer to write CSV data
    output = io.StringIO()
    # Initialize CSV writer with minimal quoting
    writer = csv.writer(output, delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)
    # Write the string as a single field in a row
    writer.writerow([s])
    # Return the CSV line, removing trailing newline characters
    return output.getvalue().rstrip('\r\n')

class AmazonReviews(MRJob):
    def mapper(self, _, line):
        yield "lines", 1
        yield get_sentiment(line), 1

    def reducer(self, key, values):
        log.info(f"Received {len(list(values))} reviews")
        log.info(f"key {key}")
        yield key, sum(values)


if __name__ == '__main__':
    AmazonReviews.run()