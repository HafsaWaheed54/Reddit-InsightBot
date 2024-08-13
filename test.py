import sqlite3

def fetch_all_tags():
    conn = sqlite3.connect('posts.db')
    c = conn.cursor()
    c.execute('SELECT tags FROM posts')
    rows = c.fetchall()
    conn.close()
    return rows

def print_tags():
    tags = fetch_all_tags()
    for tag in tags:
        print(tag[0])  # Assuming tags is the first and only element in the tuple

print_tags()
