import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
import joblib
import keras
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json
import sys

 # Load the tokenizer using joblib
with open('tokenizer.joblib', 'rb') as handle:
                tokenizer = joblib.load(handle)
                
model = keras.models.load_model('grudgehubmodelv1') 
# Define a function to predict the sentiment of input text
def predict_sentiment(text):
                # Tokenize and pad the input text
                text_sequence = tokenizer.texts_to_sequences([text])
                text_sequence = pad_sequences(text_sequence, maxlen=100)

                # Make a prediction using the trained model
                predicted_class = np.argmax(model.predict(text_sequence), axis=-1)
                if predicted_class == 0:
                    return 'Negative'
                else:
                    return 'Positive'
                
def search_google_news(query, time_range):
    # Set the base URL for Google News
    base_url = 'https://www.bing.com/news/search?q={}&qft=interval%3d"7"&form=PTFTNR'
    #base_url = 'https://www.bing.com/news/search?q={}&form=PTFTNR'

    celebName = query

    # Format the query and create the search URL
    query = query.replace(' ', '+')
    time_range = time_range.lower()
    search_url = base_url.format(query, time_range)

    # Send a GET request to the search URL
    response = requests.get(search_url)
    response.raise_for_status()
    print(search_url)

    # Create a BeautifulSoup object to parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all the news articles on the page
    article_elements = soup.find_all('div', {'class': 'news-card newsitem cardcommon'})
    #print(len(article_elements))

    # Create an instance of the SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()

    # Iterate over each article and extract relevant information
    for article in article_elements:
        # Extract the article title and link
        title_element = article.find('a', {'class': 'title'})
        title = title_element.text
        link = title_element['href']
      
        # Extract the article source and published date
        source_element = article.find('a', {'class': 'wEwyrc'})
        source = source_element.text if source_element else "Unknown Source"
        date_element = article.find('span', {'tabindex': '0'})#['datetime']
        time_ago = date_element.text
        print(date_element)
        try:
          articleImage =  'https://www.bing.com/' + article.find('img', {'role':'presentation'})['src']
        except:
            try:
              articleImage = 'https://www.bing.com/' + article.find('img', {'role':'presentation'})['data-src']
            except:
              try:
                  articleImage = 'https://www.bing.com/' + article.find('img', {'role':'presentation'})['data-src-hq']
              except:
                  articleImage = ''
        
        print(articleImage)
        
        # Convert time ago format to datetime (this is a simplified approach)
        if 'h' in time_ago:
            hours_ago = int(time_ago.split('h')[0])
            date = datetime.utcnow() - timedelta(hours=hours_ago)
        elif 'd' in time_ago:
            days_ago = int(time_ago.split('d')[0])
            date = datetime.utcnow() - timedelta(days=days_ago)
        elif 'mon' in time_ago:
            months_ago = int(time_ago.split('mon')[0])
            date = datetime.utcnow() - timedelta(days=months_ago*30)
        elif 'm' in time_ago:
            minutes_ago = int(time_ago.split('m')[0])
            date = datetime.utcnow() - timedelta(minutes=minutes_ago)
        elif 'y' in time_ago:
            years_ago = int(time_ago.split('y')[0])
            date = datetime.utcnow() - timedelta(days=years_ago * 365)
        elif 's' in time_ago:
            seconds_ago = int(time_ago.split('s')[0])
            date = datetime.utcnow() - timedelta(seconds=seconds_ago)
        

        # Check if the article is within the specified time range
        time_diff = datetime.utcnow() - date
        if time_diff <= timedelta(days=1) and celebName.lower() in title.lower():
                      
            # Predict sentiment for your own text
            text_input = title
            predicted_sentiment = predict_sentiment(text_input)
            print(predicted_sentiment)
            print(title)

            # Print the article information along with the sentiment category and score
            if predicted_sentiment == 'Negative':
                analyzer = SentimentIntensityAnalyzer()
                sentiment_score = analyzer.polarity_scores(title)['compound']
                if (sentiment_score < float(0.0)):
                    formatted_date_time = date.strftime("%Y-%m-%d %H:%M:%S")
                    print(sentiment_score)
                    print("Title:", title)
                    print("Link:", link)
                    print("Source:", source)
                    print("Published Date:",formatted_date_time)
                    print()
                    tempList = []
                    stopper = 0
                    for i in range(len(articleList)):
                        if articleList[i][1] == title.lower():
                            stopper = 1
                    if stopper == 0:
                        tempList.append(celebName.lower())
                        tempList.append(title.lower())
                        articleList.append(tempList)
                        finalLink ="https://grudge-hub.com/" + str(sys.argv[1]) + "?title="+title.replace(' ', '%20')+ "&articleImage=" + articleImage +"&date="+str(formatted_date_time).replace(' ', '%20')+"&articleText=&celebName="+query.replace(' ', '%20')+"&link="+link
                        print(finalLink)
                        requests.get(finalLink)


articleList = []
json_string = requests.get('https://grudge-hub.com/_functions/celebNames').text
parsed_json = json.loads(json_string)
names_array = parsed_json["names"]
for name in names_array:
    search_google_news(name, "y")
print(articleList)

