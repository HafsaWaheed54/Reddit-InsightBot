from flask import Flask, render_template, jsonify, request, redirect, url_for
import json
import os
import logging
from crawl import crawl_subreddit  # Import the crawl_subreddit function
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Paths to your JSON files
sysadmin_json_file = os.path.join(os.path.dirname(__file__), 'sysadmin_posts.json')
community_json_file = os.path.join(os.path.dirname(__file__), 'community_posts.json')
new_json_file = os.path.join(os.path.dirname(__file__), 'new_posts.json')

# Load the posts from all JSON files
def load_posts():
    posts = {'sysadmin': [], 'community': [], 'new': []}
    
    if os.path.exists(sysadmin_json_file):
        with open(sysadmin_json_file, 'r', encoding='utf-8') as file:
            posts['sysadmin'] = json.load(file)
    
    if os.path.exists(community_json_file):
        with open(community_json_file, 'r', encoding='utf-8') as file:
            posts['community'] = json.load(file)
    
    if os.path.exists(new_json_file):
        with open(new_json_file, 'r', encoding='utf-8') as file:
            posts['new'] = json.load(file)
    
    return posts

# Function to calculate metrics
def calculate_metrics():
    # Example data; replace with actual data as needed
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

    # Calculate metrics
    X_test = vectorizer.transform(texts)
    predictions = classifier.predict(X_test)
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }

@app.route('/')
def index():
    posts = load_posts()
    metrics = calculate_metrics()
    return render_template('index.html', sysadmin_posts=posts['sysadmin'], community_posts=posts['community'], new_posts=posts['new'], metrics=metrics)

@app.route('/api/posts')
def api_posts():
    posts = load_posts()
    return jsonify(posts)

@app.route('/crawl', methods=['POST'])
def crawl_page():
    url = request.form.get('url')
    if url:
        urls = [url]  # Wrap the single URL in a list
        try:
            logging.info(f"Crawling URL: {urls}")
            crawl_subreddit(urls)
            
            # Load newly crawled data
            with open(new_json_file, 'r', encoding='utf-8') as f:
                crawled_data = json.load(f)
            
            logging.info("Crawling completed successfully.")
            return render_template('crawl.html', posts=crawled_data)
        except Exception as e:
            logging.error(f"An error occurred while crawling: {e}")
            return f"An error occurred: {e}", 500
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
