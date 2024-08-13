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
            
            # Check for duplicate posts
            if any(post['URL'] == post_url and post['Title'] == submission.title for post in posts_list):
                logging.info(f"Duplicate post found: {submission.title} ({post_url})")
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

                # Check for duplicate sub-URLs
                if any(sub_url == url for sub_url in post['SubURLs']):
                    logging.info(f"Duplicate sub-URL found: {url}")
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
    print(f"Accuracy: {accuracy:.2f}")
    precision = precision_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions)
    
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

def get_additional_urls():
    urls = []
    print("Enter additional subreddit URLs to crawl (or type 'done' to finish):")
    while True:
        url = input("Enter URL: ").strip()
        if url.lower() == 'done':
            break
        elif url:
            urls.append(url)
        else:
            print("Please enter a valid URL or 'done' to finish.")
    return urls

def update_posts(urls):
    global forcount
    try:
        reddit = praw.Reddit(
            client_id="W2ACjJ8oo4RYrF4ui8mLjg",
            client_secret="qJ5HSyw32zeyb7mvQ5ykQnGifToWhg",
            user_agent="Chrome 126 on Windows 11"
        )

        json_file = 'sysadmin_posts.json'
        script_dir = os.path.dirname(os.path.abspath(__file__))
        json_file_path = os.path.join(script_dir, json_file)

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

        posts_list = []
        processed_urls = set()
        test_texts = texts
        test_labels = labels
        evaluate_classifier(vectorizer, classifier, test_texts, test_labels)

        if os.path.exists(json_file_path):
            with open(json_file_path, 'r', encoding='utf-8') as file:
                posts_list = json.load(file)
                logging.info(f"Existing data loaded from {json_file_path}")
        else:
            with open(json_file_path, 'w', encoding='utf-8') as file:
                json.dump(posts_list, file, ensure_ascii=False, indent=4)

        for url in urls:
            subreddit_name = extract_subreddit_name(url)
            fetch_and_save_reddit_posts(reddit, subreddit_name, posts_list, classifier, vectorizer, processed_urls)
            forcount += 1

        save_posts_to_json(posts_list, json_file_path)

    except Exception as e:
        logging.error(f"An error occurred while running the script: {e}")

if __name__ == '__main__':
    while True:
        if forcount == 0:
            update_posts([
      'https://www.reddit.com/r/sysadmin/',         'https://www.reddit.com/r/VALORANT/',         'https://www.reddit.com/r/DestinyTheGame/',         'https://www.reddit.com/r/Scams/'          'https://www.reddit.com/r/techsupport/'
      
            ])
        else:
            urls = get_additional_urls()
            if urls:
                update_posts(urls)
            else:
                logging.info("No additional URLs provided. Exiting...")
                break
        time.sleep(360)  # Run every 360 minute

