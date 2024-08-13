import os
import json
import re
import pytz
import praw
import requests
import schedule
import time
import threading
import logging
from urllib.parse import urlparse
from datetime import datetime
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import spacy

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load SpaCy model for NER
nlp = spacy.load("en_core_web_sm")

# Initial URLs to fetch
initial_urls = [
    'https://www.reddit.com/r/databreach/',
    'https://www.reddit.com/r/Databreaches/',
]

def fetch_and_save_reddit_posts(reddit, subreddit_name, posts_list, classifier, vectorizer, processed_urls, limit=None):
    try:
        logging.info(f"Fetching posts from r/{subreddit_name}...")

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
            utc_dt = datetime.utcfromtimestamp(submission.created_utc).replace(tzinfo=pytz.utc)
            pkt_dt = utc_dt.astimezone(pkt)

            post_url = submission.url.strip()
            if post_url in processed_urls:
                logging.info(f"URL already processed: {post_url}")
                continue

            processed_urls.add(post_url)

            post = {
                'Title': submission.title,
                'Author': submission.author.name if submission.author else '[deleted]',
                'Timestamp': pkt_dt.strftime('%Y-%m-%d %H:%M:%S %Z%z'),
                'URL': post_url,
                'Score': submission.score,
                'Comments': submission.num_comments,
                'Content': {'Original': submission.selftext.strip() if submission.selftext else ''},
                'CrawledTime': crawled_time.strftime('%Y-%m-%d %H:%M:%S %Z%z'),
                'SubURL': '',
                'Email': '',
                'SubContent': '',
                'Entities': [],
                'SubEntities': [],
                'Category': 'Unknown',
                'SubCategory': 'Unknown',
                'Tag': 'not breach',
                'SubTag': 'not breach'
            }

            # Extract and clean URLs
            urls = re.findall(r'http[s]?://[^\s\)]+', post['Content']['Original'])
            if urls:
                post['SubURL'] = urls[0].rstrip(')')  # Remove trailing parentheses

                # Ensure SubURL is valid
                if urlparse(post['SubURL']).scheme and urlparse(post['SubURL']).netloc:
                    try:
                        logging.info(f"Fetching content from URL: {post['SubURL']}")
                        response = requests.get(post['SubURL'], allow_redirects=True, timeout=10)
                        response.raise_for_status()  # Ensure we catch HTTP errors

                        # Check if the content type is text
                        if 'text' in response.headers.get('Content-Type', ''):
                            response.encoding = response.apparent_encoding  # Set encoding to apparent encoding
                            soup = BeautifulSoup(response.content, 'html.parser')

                            # Extract emails and content
                            post['Email'] = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', soup.get_text())
                            post['SubContent'] = soup.get_text()
                            doc = nlp(post['SubContent'])
                            post['SubEntities'] = [(ent.text, ent.label_) for ent in doc.ents]

                            # Predict category and tag for the content
                            sub_content = post['SubContent'].lower()
                            if sub_content:
                                for category, keyword_list in keywords.items():
                                    if any(keyword in sub_content for keyword in keyword_list):
                                        post['SubCategory'] = category
                                        break
                                else:
                                    post['SubCategory'] = 'Unknown'

                                X_sub = vectorizer.transform([sub_content])
                                sub_prediction = classifier.predict(X_sub)
                                post['SubTag'] = 'breach' if sub_prediction[0] else 'not breach'
                        else:
                            logging.warning(f"Content at {post['SubURL']} is not text-based. Skipping content extraction.")
                    except requests.RequestException as e:
                        logging.error(f"Failed to crawl {post['SubURL']}: {e}")
                    except Exception as e:
                        logging.error(f"An error occurred while processing the URL content: {e}")
                else:
                    logging.error(f"Invalid URL format: {post['SubURL']}")

            # Categorize the post content
            if post['Content']['Original']:
                content = post['Content']['Original'].lower()
                for category, keyword_list in keywords.items():
                    if any(keyword in content for keyword in keyword_list):
                        post['Category'] = category
                        break
                else:
                    post['Category'] = 'Unknown'

                logging.info(f"Post categorized as {post['Category']}: {post['Title']}")

                doc = nlp(post['Content']['Original'])
                post['Entities'] = [(ent.text, ent.label_) for ent in doc.ents]

                X = vectorizer.transform([content])
                prediction = classifier.predict(X)
                post['Tag'] = 'breach' if prediction[0] else 'not breach'

            posts_list.append(post)

        logging.info(f"Posts from r/{subreddit_name} added to the list.")

    except Exception as e:
        logging.error(f"An error occurred while fetching posts: {e}")

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

def get_additional_urls():
    urls = initial_urls.copy()  # Start with initial URLs
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
    try:
        logging.info("Setting up Reddit API credentials...")
        reddit = praw.Reddit(
            client_id="W2ACjJ8oo4RYrF4ui8mLjg",
            client_secret="qJ5HSyw32zeyb7mvQ5ykQnGifToWhg",
            user_agent="Chrome 126 on Windows 11"
        )

        json_file = 'community_posts.json'
        script_dir = os.path.dirname(os.path.abspath(__file__))
        json_file_path = os.path.join(script_dir, json_file)
        logging.info(f"Saving to {json_file_path}")

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

        all_posts_list = []
        processed_urls = set()

        for url in urls:
            try:
                subreddit_name = extract_subreddit_name(url)
                fetch_and_save_reddit_posts(
                    reddit=reddit,
                    subreddit_name=subreddit_name,
                    posts_list=all_posts_list,
                    classifier=classifier,
                    vectorizer=vectorizer,
                    processed_urls=processed_urls,
                )
            except ValueError as ve:
                logging.error(f"ValueError: {ve}")
            except Exception as e:
                logging.error(f"An error occurred while processing URL {url}: {e}")

        save_posts_to_json(all_posts_list, json_file_path)

    except Exception as e:
        logging.error(f"An error occurred: {e}")

def schedule_updates(urls):
    schedule.every(6).hours.do(update_posts, urls=urls)
    while True:
        schedule.run_pending()
        time.sleep(1)

def main():
    urls = initial_urls.copy()  # Start with the default URLs
    update_posts(urls)  # Fetch the default URLs initially

    # Optionally, start scheduling updates
    print("Would you like to schedule periodic updates? (yes/no): ")
    if input().strip().lower() == 'yes':
        schedule_thread = threading.Thread(target=schedule_updates, args=(urls,))
        schedule_thread.start()

    # Prompt for additional URLs
    additional_urls = get_additional_urls()
    if additional_urls:
        update_posts(additional_urls)  # Fetch additional URLs

if __name__ == "__main__":
    main()
