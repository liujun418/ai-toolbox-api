"""Blog post sync tool — create/update posts via direct PostgreSQL.

Usage:
  python scripts/blog_sync.py              # List all posts
  python scripts/blog_sync.py --add slug   # Add from template
  python scripts/blog_sync.py --delete slug # Delete a post

Uses direct DB connection (same as seed_blog_direct.py).
"""

import os
import sys
import uuid
from datetime import datetime, timezone
import psycopg2

DB = os.getenv("DATABASE_PUBLIC_URL")
if not DB:
    raise RuntimeError("DATABASE_PUBLIC_URL environment variable is required")

def get_conn():
    return psycopg2.connect(DB)

def list_posts():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('SELECT slug, title, category, published, created_at FROM blog_posts ORDER BY created_at DESC')
    for row in cur.fetchall():
        print(f'  [{row[2]}] {row[0]:50s} | {row[1][:70]} {"(draft)" if not row[3] else ""}')
    print(f'  --- {cur.rowcount} total ---')
    conn.close()

def add_post(slug):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('SELECT id FROM blog_posts WHERE slug = %s', (slug,))
    if cur.fetchone():
        print(f'Post "{slug}" already exists. Use --update instead.')
        conn.close()
        return

    now = datetime.now(timezone.utc).isoformat()
    pid = str(uuid.uuid4())
    cur.execute(
        'INSERT INTO blog_posts (id, slug, title, description, content, category, tags, related_tools, published, created_at, updated_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,TRUE,%s,%s)',
        (pid, slug, 'New Post', 'Description here', '<p>Content here</p>', 'General', '', None, now, now)
    )
    conn.commit()
    conn.close()
    print(f'Created draft: {slug} (edit in admin panel)')

def delete_post(slug):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('DELETE FROM blog_posts WHERE slug = %s', (slug,))
    conn.commit()
    deleted = cur.rowcount
    conn.close()
    print(f'Deleted: {slug}' if deleted else f'Not found: {slug}')

if __name__ == '__main__':
    if '--add' in sys.argv:
        idx = sys.argv.index('--add')
        add_post(sys.argv[idx + 1] if idx + 1 < len(sys.argv) else 'new-post')
    elif '--delete' in sys.argv:
        idx = sys.argv.index('--delete')
        delete_post(sys.argv[idx + 1] if idx + 1 < len(sys.argv) else '')
    else:
        list_posts()
