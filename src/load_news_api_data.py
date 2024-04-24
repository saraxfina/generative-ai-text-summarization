small_dataset = False
verbose = False

import os
import requests
from newspaper import Article
import sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

api_key = os.environ['NEWS_API_KEY']
print(api_key)

url = 'https://newsapi.org/v2/everything'
country = 'us'
pageSize = '100'
page = '1'
from_date = '2024-04-20'
to_date = '2024-04-22'
domains = 'cnn.com,bbc.co.uk,nbc.com,nprnews.com,foxnews.com,washingtonpost.com,reuters.com,nytimes.com'
excludeDomains = 'cnnespanol.cnn.com,arabic.cnn.com'
sortBy = 'popularity'
request = f'%s?domains=%s&excludeDomains=%s&from=%s&to=%s&sortBy=%s&pageSize=%s&page=%s&apiKey=%s' % (url, domains, excludeDomains, from_date, to_date, sortBy, pageSize, page, api_key)

print(request)

response = requests.get(request)

print(response.json()['totalResults'])

data = response.json()['articles']

dataset = []
count = 0

if small_dataset:
    max_articles = 10
else:
    max_articles = 100
    

for article in data:
    url = article['url']

    # remove articles that NewsAPI did not return
    if url == "https://removed.com":
        continue 
        
    # remove foreign articles 
    if "espanol" in url or "arabic" in url:
        continue
        
    # remove articles that can't be scraped
    news_article = Article(url)
    try: 
        news_article.download()
        news_article.parse()
    except:
        print("failed to download")
        continue 
    
    content = news_article.text
    
    # remove articles that have a length of 0 
    if len(content) == 0 or len(article['description']) == 0:
        continue
        
    if len(content) < 500:
        continue
        
    if verbose:
        print(str(count) + " " + url)
        print("ARTICLE: " + content)
        print("HIGHLIGHT: " + article['description'])
    
    
    # save all other articles 
    dataset.append({'article' : content, 'highlights' : article['description']} )
    count += 1
    
    if count == max_articles: 
        break

data = pd.DataFrame.from_dict(dataset)

train, test = train_test_split(data, test_size=.2, shuffle=False, random_state=2)
train, val = train_test_split(train, test_size=.25, shuffle=False, random_state=2)

X_train = train.iloc[:,0]
y_train = train.iloc[:,1]
X_test = test.iloc[:,0]
y_test = test.iloc[:,1]
X_val = val.iloc[:,0]
y_val = val.iloc[:,1]

if small_dataset:
    np.savez('../data/news_api_data_small.npz' ,  X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, X_val=X_val, y_val=y_val)
else:
    np.savez('../data/news_api_data.npz' ,  X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, X_val=X_val, y_val=y_val)
