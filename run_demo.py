#!/usr/bin/env python3
"""
Automated Demonstration Script for Test - Set A.
Executes end-to-end demonstrations for both:
  - Q1: Automatic Document Categorization System
  - Q2: Multimedia Search and Retrieval System
"""

import os
import sys
from src.q1_categorization import DocumentCategorizer
from src.q2_multimedia import MultimediaRetrievalSystem
from src.utils import initialize_sample_dataset, load_documents_from_directory


def header(title):
    print("\n" + "=" * 80)
    print(f" {title.upper()}")
    print("=" * 80)


def subheader(title):
    print("\n" + "-" * 80)
    print(f" >>> {title}")
    print("-" * 80)


def run_q1_demo():
    header("Demonstration of Q1: Automatic Document Categorization System")

    docs_dir, _, _ = initialize_sample_dataset(".")
    docs = load_documents_from_directory(docs_dir)
    print(f"Loaded {len(docs)} sample documents across 5 diverse subject domains.")

    # 1. K-Means
    subheader("1. Converting to TF-IDF & Clustering with K-Means (K=5)")
    categorizer = DocumentCategorizer()
    categorizer.fit(docs)
    categorizer.cluster(method="kmeans", n_clusters=5, random_state=42)

    groups = categorizer.get_clustered_groups()
    for g in groups:
        top_words = ", ".join([f"{w} ({s:.2f})" for w, s in g["top_keywords"][:4]])
        print(f"\n📁 [{g['cluster_name']}] (Docs: {g['doc_count']})")
        print(f"   Significant Features: {top_words}")
        for d in g["documents"]:
            print(f"     • {d['title']} (Centroid Sim: {d['similarity_to_centroid']:.4f})")

    # 2. Hierarchical Clustering comparison
    subheader("2. Alternative Clustering: Hierarchical Agglomerative Clustering (K=5)")
    h_categorizer = DocumentCategorizer()
    h_categorizer.fit(docs)
    h_categorizer.cluster(method="hierarchical", n_clusters=5)
    h_groups = h_categorizer.get_clustered_groups()
    for g in h_groups:
        print(f"   • {g['cluster_name']} -> {g['doc_count']} docs ({', '.join([d['title'] for d in g['documents']])})")

    # 3. Query relevance testing
    queries = [
        "artificial intelligence neural network backpropagation gpu",
        "cardiovascular disease blood pressure cholesterol heart",
        "james webb telescope deep space galaxies and dark matter",
    ]

    for q in queries:
        subheader(f"3. User Query Matching: \"{q}\"")
        res = categorizer.query(q, top_k_docs=3)
        print(f"🎯 Most Relevant Cluster: {res['best_cluster_name']} (Score: {res['best_cluster_score']:.4f})")
        print("\n📄 Top Relevant Documents:")
        for d in res["ranked_documents"]:
            print(f"   #{d['rank']} [{d['relevance_score']:.4f}] {d['title']} (Cluster: {d['cluster_name']})")
            print(f"       Excerpt: {d['snippet'][:120]}...")


def run_q2_demo():
    header("Demonstration of Q2: Multimedia Search and Retrieval System")

    _, _, metadata_path = initialize_sample_dataset(".")
    system = MultimediaRetrievalSystem(metadata_path, base_dir=".")
    print(f"Loaded multimedia repository with {len(system.items)} items (Text Docs & Images).")

    # 1. Metadata keyword search
    text_queries = ["coronary artery cardiology", "ocean marine water", "blockchain cryptocurrency fintech"]
    for q in text_queries:
        subheader(f"1. Keyword Search over Metadata: \"{q}\"")
        results = system.search_metadata(q, top_k=3)
        for rank, r in enumerate(results, start=1):
            media_badge = "[TEXT]" if r["media_type"] == "text" else "[IMAGE]"
            print(f"   #{rank} {media_badge} {r['title']} (Score: {r['relevance_score']:.4f})")
            print(f"       Category: {r['category']} | Uploaded: {r['upload_date']} | Path: {r['file_path']}")
            print(f"       Description: {r['description'][:100]}...")

    # 2. Image retrieval via color histograms
    image_queries = [
        ("data/queries/query_green_leaf.png", "Green Botanical Leaf Specimen"),
        ("data/queries/query_blue_water.png", "Blue Lagoon Aquatic Water"),
        ("data/queries/query_red_fire.png", "Glowing Crimson Campfire Embers"),
    ]

    for qpath, label in image_queries:
        subheader(f"2. Image Retrieval via 3D/Channel Color Histogram: {label}")
        results = system.retrieve_similar_images(qpath, top_k=3, metric="intersection")
        for rank, r in enumerate(results, start=1):
            print(f"   #{rank} [IMAGE] {r['title']} (Histogram Similarity: {r['relevance_score']:.4f})")
            print(f"       File: {r['file_path']} | Category: {r['category']}")

    # 3. Multimodal combined search
    subheader("3. Multimodal Unified Search (Keyword + Image)")
    fused_results = system.unified_search(
        query_text="deep space cosmic nebula astronomy",
        query_image_input="data/images/galaxy_neon_purple.png",
        top_k=4,
    )
    for rank, r in enumerate(fused_results, start=1):
        media_badge = "[TEXT]" if r["media_type"] == "text" else "[IMAGE]"
        print(f"   #{rank} {media_badge} {r['title']} (Fused Score: {r['relevance_score']:.4f})")
        print(f"       Category: {r['category']} | File: {r['file_path']}")


def main():
    print("=" * 80)
    print("  INFORMATION RETRIEVAL SYSTEMS (IRS) - ASSIGNMENT TEST SET A")
    print("  Automated Verification and Showcase Script")
    print("=" * 80)

    run_q1_demo()
    run_q2_demo()

    print("\n" + "=" * 80)
    print("  ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
