import psycopg2
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import requests
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host":     os.getenv("PG_HOST", "10.6.21.3"),
    "port":     int(os.getenv("PG_PORT", 5432)),
    "dbname":   os.getenv("PG_DB",   "optimize"),
    "user":     os.getenv("PG_USER", "kietcorn"),
    "password": os.getenv("PG_PASS", "kiietqo9204"),
    "options":  "-c client_encoding=UTF8",
}

def fetch_title_embeddings():
    print("Connecting to database...")
    conn = psycopg2.connect(**DB_CONFIG)
    rows = []
    seen_titles = set()

    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, text, embedding::text
            FROM n8n_vectors
            WHERE embedding IS NOT NULL
        """)
        for row_id, text, emb_text in cur.fetchall():
            if not text.startswith("Title:"):
                continue
            if text in seen_titles:
                continue
            seen_titles.add(text)
            vector = json.loads(emb_text)
            rows.append({"id": str(row_id), "title": text, "vector": vector})

    conn.close()
    print(f"Loaded {len(rows)} unique titles.")
    return rows
rows = fetch_title_embeddings()
print (len(rows))