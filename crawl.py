from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os
import json
import re
import pytz
import praw
import requests
import logging
import schedule
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from urllib.parse import urlparse
from datetime import datetime
from bs4 import BeautifulSoup
import spacy

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load SpaCy model for NER
nlp = spacy.load("en_core_web_sm")
forcount = 0

def fetch_and_save_reddit_posts(reddit, subreddit_name, posts_list, classifier, vectorizer, processed_urls, limit=30):
    try:
        subreddit = reddit.subreddit(subreddit_name)
        pkt = pytz.timezone('Asia/Karachi')
        crawled_time = datetime.now(pytz.utc).astimezone(pkt)

        keywords = {
            'recommendation': ["i just", "I just", "However", "I got", "following", "guide"],
            'url': ["http://", "https://"],
            'question': ["how", "sell?", "?", "idea", "What", "what", "What are"],
            'conversation': ["hello", "hey", "anyone", "Hi", "Many", "hi"],
            'announcement': ["announcement", "news", "update"],
            'tutorial': ["how to", "tutorial", "guide", "step by step"],
            'discussion': ["what do you think", "thoughts", "discuss", "opinions"]
        }

        for submission in subreddit.new(limit=limit):
            utc_dt = datetime.fromtimestamp(submission.created_utc, tz=pytz.utc)
            pkt_dt = utc_dt.astimezone(pkt)

            post_url = clean_url(submission.url)
            if process_url(post_url, processed_urls):
                continue
            content = submission.selftext.strip() if submission.selftext else ''
            if not content:
                # Fetch content from linked URL if submission is a link
                try:
                    response = requests.get(post_url, allow_redirects=True)
                    content_type = response.headers.get('Content-Type', '')
                    if 'text/html' in content_type:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        content = soup.get_text()
                    else:
                        logging.info(f"URL points to non-text content (Content-Type: {content_type}). Skipping...")
                except requests.RequestException as e:
                    logging.error(f"Error fetching content from {post_url}: {e}")
                    content = ''

            post = {
                'Title': submission.title,
                'Author': submission.author.name if submission.author else '[deleted]',
                'Timestamp': pkt_dt.strftime('%Y-%m-%d %H:%M:%S %Z%z'),
                'URL': post_url,
                'Score': submission.score,
                'Comments': submission.num_comments,
                'Content': {'Original': content},
                'CrawledTime': crawled_time.strftime('%Y-%m-%d %H:%M:%S %Z%z'),
                'SubURLs': [],
                'SubContents': [],
                'SubEntities': [],
                'Categories': [],
                'Tags': [],
                'Emails': []
            }

            # Handle URLs in content
            urls = re.findall(r'\((http[s]?://\S+)\)', post['Content']['Original'])
            for url in urls:
                url = clean_url(url)
                if process_url(url, processed_urls):
                    continue

                post['SubURLs'].append(url)
                try:
                    response = requests.get(url, allow_redirects=True)
                    content_type = response.headers.get('Content-Type', '')
                    if 'text/html' in content_type:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        sub_content = soup.get_text()
                        post['Emails'].append(re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', sub_content))
                        post['SubContents'].append({'SubURL': url, 'Content': sub_content})
                        doc = nlp(sub_content)
                        post['SubEntities'].append([(ent.text, ent.label_) for ent in doc.ents])

                        sub_content_lower = sub_content.lower()
                        if sub_content_lower:
                            for category, keyword_list in keywords.items():
                                if any(keyword in sub_content_lower for keyword in keyword_list):
                                    post['Categories'].append(category)
                                    break
                            else:
                                post['Categories'].append('Unknown')

                            X_sub = vectorizer.transform([sub_content_lower])
                            sub_prediction = classifier.predict(X_sub)
                            post['Tags'].append('breach' if sub_prediction[0] else 'not breach')
                    else:
                        logging.info(f"URL points to non-text content (Content-Type: {content_type}). Skipping...")
                except requests.RequestException as e:
                    logging.error(f"Error fetching content from {url}: {e}")
                except Exception as e:
                    logging.error(f"Error processing URL {url}: {e}")

            if post['Content']['Original']:
                content = post['Content']['Original'].lower()
                for category, keyword_list in keywords.items():
                    if any(keyword in content for keyword in keyword_list):
                        post['Categories'].append(category)
                        break
                else:
                    post['Categories'].append('Unknown')

                doc = nlp(post['Content']['Original'])
                post['Entities'] = [(ent.text, ent.label_) for ent in doc.ents]

                X = vectorizer.transform([content])
                prediction = classifier.predict(X)
                post['Tags'].append('breach' if prediction[0] else 'not breach')

            posts_list.append(post)

        logging.info(f"Posts from r/{subreddit_name} added to the list.")

    except Exception as e:
        logging.error(f"An error occurred while fetching posts from r/{subreddit_name}: {e}")


def clean_url(url):
    """ Clean the URL by removing unwanted characters. """
    url = url.strip()
    url = re.sub(r'[^\w\s:/\.\-]', '', url)
    return url


def process_url(url, processed_urls):
    """ Check if URL has been processed and add to set if not. """
    if url in processed_urls:
        logging.info(f"URL already processed: {url}")
        return True
    processed_urls.add(url)
    return False


def extract_subreddit_name(url):
    parsed_url = urlparse(url)
    path_parts = parsed_url.path.strip('/').split('/')
    if len(path_parts) > 1 and path_parts[0].lower() == 'r':
        return path_parts[1]
    else:
        raise ValueError("Invalid subreddit URL")

def save_posts_to_json(posts_list, json_filename):
    try:
        with open(json_filename, 'w', encoding='utf-8') as json_file:
            json.dump(posts_list, json_file, ensure_ascii=False, indent=4)
        logging.info(f"All posts saved to {json_filename}")
    except Exception as e:
        logging.error(f"An error occurred while saving the file: {e}")

def evaluate_classifier(vectorizer, classifier, test_texts, test_labels):
    """ Evaluate and print the accuracy of the classifier. """
    X_test = vectorizer.transform(test_texts)
    predictions = classifier.predict(X_test)
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Classifier accuracy: {accuracy:.2f}")
    precision = precision_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions)
    
    print(f"Classifier precision: {precision:.2f}")
    print(f"Classifier recall: {recall:.2f}")


    
def crawl_subreddit(urls):
    """ Crawl subreddit based on list of URLs. """
    try:
        reddit = praw.Reddit(
          client_id="W2ACjJ8oo4RYrF4ui8mLjg",
            client_secret="qJ5HSyw32zeyb7mvQ5ykQnGifToWhg",
            user_agent="Chrome 126 on Windows 11"
        )

        # Initialize the vectorizer and classifier
        breach_texts = [
            "data breach", "leak", "hacked", "compromised", "exposed", "cyber attack", "ransomware", "Breach", "breach"
        ]
        not_breach_texts = [
            "update", "patch", "new feature", "service announcement", "policy change"
        ]
        texts = breach_texts + not_breach_texts
        labels = [1] * len(breach_texts) + [0] * len(not_breach_texts)

        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(texts)
        classifier = LogisticRegression()
        classifier.fit(X, labels)

        evaluate_classifier(vectorizer, classifier, texts, labels)

        posts_list = []
        processed_urls = set()

        for url in urls:
            print(url)
            try:
                subreddit_name = extract_subreddit_name(url)
                fetch_and_save_reddit_posts(reddit, subreddit_name, posts_list, classifier, vectorizer, processed_urls)
            except ValueError as ve:
                logging.error(f"ValueError: {ve}")
            except Exception as e:
                logging.error(f"An error occurred while processing URL {url}: {e}")

        new_json_file = os.path.join(os.path.dirname(__file__), 'new_posts.json')
        with open(new_json_file, 'w', encoding='utf-8') as f:
            json.dump(posts_list, f, ensure_ascii=False, indent=4)
        logging.info(f"Posts saved to {new_json_file}")

    except Exception as e:
        logging.error(f"An error occurred while updating posts: {e}")
