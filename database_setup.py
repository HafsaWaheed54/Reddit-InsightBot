import sqlite3
import json
import spacy
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load SpaCy model for NER
nlp = spacy.load("en_core_web_sm")

# Database setup
def create_table_if_not_exists():
    conn = sqlite3.connect('posts.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            author TEXT,
            timestamp TEXT,
            crawled_time TEXT,
            url TEXT,
            email TEXT,
            sub_content TEXT,
            category TEXT,
            tags TEXT,
            content TEXT,
            entities TEXT,
            file_content BLOB
        )
    ''')
    conn.commit()
    conn.close()

def read_file_content(file_path):
    with open(file_path, 'rb') as file:
        return file.read()

def insert_post_with_file_content(post, file_path):
    file_content = read_file_content(file_path)
    conn = sqlite3.connect('posts.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO posts (title, author, timestamp, crawled_time, url, email, sub_content, category, tags, content, entities, file_content)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        post.get('Title', ''),
        post.get('Author', ''),
        post.get('Timestamp', ''),
        post.get('CrawledTime', ''),
        post.get('SubURL', ''),
        post.get('Email', ''),
        post.get('SubContent', ''),
        post.get('Category', ''),
        post.get('Tag', ''),
        json.dumps(post.get('Content', {})) if post.get('Content') is not None else '{}',
        json.dumps(post.get('Entities', []), ensure_ascii=False) if post.get('Entities') is not None else '[]',
        file_content
    ))
    conn.commit()
    conn.close()

def fetch_file_content_from_db(post_id):
    conn = sqlite3.connect('posts.db')
    c = conn.cursor()
    c.execute('SELECT file_content FROM posts WHERE id = ?', (post_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return row[0]
    return None

def fetch_posts_from_db():
    conn = sqlite3.connect('posts.db')
    c = conn.cursor()
    c.execute('SELECT id, title, author, timestamp, crawled_time, url, email, sub_content, category, tags, content, entities FROM posts')
    rows = c.fetchall()
    posts = []
    for row in rows:
        posts.append({
            'ID': row[0],
            'Title': row[1],
            'Author': row[2],
            'Timestamp': row[3],
            'CrawledTime': row[4],
            'SubURL': row[5],
            'Email': row[6],
            'SubContent': row[7],
            'Category': row[8],
            'Tag': row[9],
            'Content': json.loads(row[10]) if row[10] else {},
            'Entities': json.loads(row[11]) if row[11] else []
        })
    conn.close()
    return posts

def process_posts_from_file(json_filename):
    try:
        with open(json_filename, 'r', encoding='utf-8') as json_file:
            posts = json.load(json_file)

        for post in posts:
            if 'Content' in post and post['Content']:
                # Apply NER to the content
                doc = nlp(post['Content'].get('Original', ''))
                entities = [(ent.text, ent.label_) for ent in doc.ents]
                post['Entities'] = entities

                # Update post in the database
                conn = sqlite3.connect('posts.db')
                c = conn.cursor()
                c.execute('''
                    UPDATE posts
                    SET entities = ?
                    WHERE title = ?
                ''', (json.dumps(entities), post.get('Title', '')))
                conn.commit()
                conn.close()

        logging.info(f"Processed posts from {json_filename}")

    except Exception as e:
        logging.error(f"An error occurred while processing the file: {e}")

# Ensure the table exists
create_table_if_not_exists()

# Process posts from the JSON file
process_posts_from_file('community_posts.json')
process_posts_from_file('sysadmin_posts.json')
