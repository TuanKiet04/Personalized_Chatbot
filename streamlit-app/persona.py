import os
import json
import csv
import numpy as np
import requests
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "models/gemini-2.5-flash-lite"
GEMINI_URL   = f"https://generativelanguage.googleapis.com/v1beta/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
OLLAMA_URL     = "http://localhost:11435/api/embeddings"
EMBED_MODEL    = "nomic-embed-text:latest"
CSV_PATH       = "raw_data.csv"


def load_articles(csv_path):
    articles = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("content") and row.get("title"):
                articles.append({
                    "id":      row["id"],
                    "title":   row["title"],
                    "content": row["content"],
                    "topic":   row.get("topic", ""),
                    "url":     row.get("url", ""),
                })
    print(f"Loaded {len(articles)} articles from CSV.")
    return articles


def embed_texts(texts):
    vectors = []
    for i, text in enumerate(texts):
        response = requests.post(OLLAMA_URL, json={
            "model": EMBED_MODEL,
            "prompt": text
        })
        response.raise_for_status()
        vectors.append(response.json()["embedding"])
        if (i + 1) % 10 == 0:
            print(f"  Embedded {i + 1}/{len(texts)}")
    return np.array(vectors, dtype=np.float32)


def find_optimal_k(X, k_min=2, k_max=10):
    print(f"\nEvaluating K from {k_min} to {k_max}...")
    silhouettes = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, init="k-means++", max_iter=300, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels, sample_size=min(1000, len(X)), random_state=42)
        silhouettes.append(sil)
        print(f"  K={k:2d} | Silhouette={sil:.4f}")
    best_idx = int(np.argmax(silhouettes))
    optimal_k = k_min + best_idx
    print(f"\nOptimal K={optimal_k} (Silhouette={max(silhouettes):.4f})")
    return optimal_k, silhouettes


def plot_silhouette(silhouettes, k_min, k_max, optimal_k):
    k_range = list(range(k_min, k_max + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, silhouettes, "bs-", linewidth=2, markersize=6)
    plt.axvline(x=optimal_k, color="r", linestyle="--", label=f"K optimal = {optimal_k}")
    plt.title("Silhouette Score by number of clusters")
    plt.xlabel("K (number of Personas)")
    plt.ylabel("Silhouette Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("silhouette.png", dpi=300, bbox_inches="tight")
    print("Saved: silhouette.png")
    plt.show()


def cluster_articles(articles, X, optimal_k):
    km = KMeans(n_clusters=optimal_k, init="k-means++", max_iter=300, n_init=10, random_state=42)
    labels = km.fit_predict(X)

    clusters = {i: [] for i in range(optimal_k)}
    for idx, label in enumerate(labels):
        clusters[label].append(articles[idx])

    representatives = {}
    for cluster_id, members in clusters.items():
        centroid = km.cluster_centers_[cluster_id]
        indices  = [articles.index(m) for m in members]
        vecs     = X[indices]
        dists    = np.linalg.norm(vecs - centroid, axis=1)
        sorted_members = [members[i] for i in np.argsort(dists)]
        representatives[cluster_id] = sorted_members[:5]

    return clusters, representatives


def call_gemini(prompt):
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 512}
    }
    response = requests.post(GEMINI_URL, json=payload)
    response.raise_for_status()
    return response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


def generate_personas(representatives):
    personas = {}
    print("\nGenerating Persona prompts via Gemini...")
    for cluster_id, members in representatives.items():
        titles = "\n".join([f"- {m['title']}" for m in members])
        prompt = f"""Dưới đây là một số tiêu đề bài báo đại diện cho một nhóm độc giả:

{titles}

Dựa trên các chủ đề trên, hãy tạo một Persona ngắn gọn cho nhóm độc giả này gồm:
1. Tên Persona (ví dụ: "Nhà đầu tư thận trọng", "Người quan tâm sức khỏe"...)
2. Mô tả ngắn về họ là ai, quan tâm gì (2-3 câu)
3. System prompt cho chatbot để giao tiếp phù hợp với nhóm này (3-5 câu, viết theo góc độ hướng dẫn chatbot)

Trả lời theo định dạng JSON với các trường: name, description, system_prompt"""

        print(f"  Generating Persona for cluster {cluster_id}...")
        try:
            raw   = call_gemini(prompt)
            clean = raw.replace("```json", "").replace("```", "").strip()
            persona = json.loads(clean)
        except Exception as e:
            print(f"  Warning: cluster {cluster_id}: {e}")
            persona = {"name": f"Persona {cluster_id}", "description": "", "system_prompt": ""}
        personas[cluster_id] = {
            "persona": persona,
            "representative_titles": [m["title"] for m in members],
        }
    return personas


def save_results(clusters, personas, output_path="personas.json"):
    output = []
    for cluster_id, persona_data in personas.items():
        all_titles = [m["title"] for m in clusters[cluster_id]]
        output.append({
            "cluster_id":            cluster_id,
            "persona":               persona_data["persona"],
            "representative_titles": persona_data["representative_titles"],
            "all_titles":            all_titles,
            "total_articles":        len(all_titles),
        })
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {output_path}")
    return output


if __name__ == "__main__":
    articles = load_articles(CSV_PATH)

    print("\nEmbedding articles via Ollama...")
    texts = [f"{a['title']}\n{a['content']}" for a in articles]
    X = embed_texts(texts)
    print(f"Embedding done. Shape: {X.shape}")

    K_MIN, K_MAX = 2, 10
    optimal_k, silhouettes = find_optimal_k(X, K_MIN, K_MAX)
    plot_silhouette(silhouettes, K_MIN, K_MAX, optimal_k)

    clusters, representatives = cluster_articles(articles, X, optimal_k)
    personas = generate_personas(representatives)
    result   = save_results(clusters, personas)

    print("\nDone. Summary:")
    for item in result:
        print(f"  Cluster {item['cluster_id']}: {item['persona'].get('name', 'N/A')} ({item['total_articles']} articles)")