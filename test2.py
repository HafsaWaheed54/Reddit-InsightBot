import sqlite3

def fetch_all_posts():
    conn = sqlite3.connect('posts.db')
    c = conn.cursor()
    # Adjust the SQL query to select the content of the posts
    c.execute('SELECT content FROM posts')
    rows = c.fetchall()
    conn.close()
    return rows

def print_posts():
    posts = fetch_all_posts()
    for post in posts:
        print(post[0])  # Assuming content is the first and only element in the tuple

print_posts()
