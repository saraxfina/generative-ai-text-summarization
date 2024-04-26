# docker run -p 8000:8000 capstone

import requests
from newspaper import Article
import argparse
import re
import json

parser = argparse.ArgumentParser("demo")
parser.add_argument("url", help="Provide a url to a news article")
args = parser.parse_args() 
   
news_article = Article(args.url)

if args.url == "authors":
    
    addr = 'http://127.0.0.1:8000/authors'
    headers = {'Content-Type' : 'application/json' }
    r = requests.get(addr, headers=headers)

    response = json.loads(r.text)
    print("RESPONSE: " + response['message']) 
    exit(0)


print("URL: " + args.url)
try: 
    news_article.download()
    news_article.parse()
except:
    print("ERROR: Article cannot be scraped")
    exit(1)  

article = [news_article.text]

if len(article) == 0:
    print("ERROR: Article download issue")
    exit(1)

article = [re.sub(r'\r\n|\r|\n', ' ', _) for _ in article]
article = [re.sub(r"[-]", ' ', _) for _ in article] # replace dashes w/ space
article = [re.sub(' +', ' ', _) for _ in article] # replace multiple spaces w/ single
article  = [re.sub('[^a-zA-Z0-9 ]+', '', _) for _ in article]

print()
print("Article Length: " + str(len(article[0].split())))
print() 

article = article[0]

addr = 'http://127.0.0.1:8000/predict'
headers = {'Content-Type' : 'application/json' }
payload = { 'input_text' : article  }
payload = json.dumps(payload)
r = requests.post(addr, data=payload, headers=headers)

response = json.loads(r.text)
print("SUMMARY: " + response['summary']) 

