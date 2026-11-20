#!/usr/bin/env python3
"""
CLI Application for Q1: Automatic Document Categorization System.

Usage:
    python3 q1_cli.py --query "artificial intelligence deep learning"
    python3 q1_cli.py --method hierarchical --n-clusters 5
    python3 q1_cli.py --interactive
"""

import argparse
import os
import sys
from src.q1_categorization import DocumentCategorizer
from src.utils import initialize_sample_dataset, load_documents_from_directory


def format_separator(char="=", length=80):
    return char * length


def print_clusters(groups):
    print("\n" + format_separator("="))
    print(f"DOCUMENT CLUSTERING RESULTS ({len(groups)} Clusters Found)")
    print(format_separator("="))

    for g in groups:
        k_num = g["cluster_number"]
        name = g["cluster_name"]
        count = g["doc_count"]
        top_words = ", ".join([f"{term} ({wt:.3f})" for term, wt in g["top_keywords"][:5]])

        print(f"\n📂 [{name}] — {count} Document(s)")
        print(f"   Top TF-IDF Terms: {top_words}")
        print("   Documents:")
        for doc in g["documents"]:
            sim = doc["similarity_to_centroid"]
            print(f"     • [{doc['id']}] {doc['title']} (Centroid Sim: {sim:.4f})")
            print(f"       Summary: {doc['snippet'][:110]}...")


def print_query_results(result):
    print("\n" + format_separator("-"))
    print(f"QUERY RELEVANCE RESULTS: \"{result['query']}\"")
    print(format_separator("-"))

    if result["status"] == "out_of_vocabulary":
        print("⚠️  Warning: None of the query terms were present in document vocabulary.")
        return

    best_cid = result["best_cluster_id"]
    best_cname = result["best_cluster_name"]
    best_score = result["best_cluster_score"]

    print(f"\n🎯 MOST RELEVANT CLUSTER:")
    print(f"   • Cluster: {best_cname}")
    print(f"   • Relevance Score: {best_score:.4f}")

    print("\n📊 Cluster Similarity Distribution:")
    for cid, cinfo in result["cluster_scores"].items():
        marker = "👉 " if cid == best_cid else "   "
        print(f" {marker}{cinfo['cluster_name']}: {cinfo['score']:.4f}")

    print("\n📄 MOST RELEVANT DOCUMENTS:")
    ranked_docs = result["ranked_documents"]
    if not ranked_docs:
        print("   No matching documents found.")
    else:
        for doc in ranked_docs:
            cluster_tag = "⭐ [In Best Cluster]" if doc["is_in_best_cluster"] else "   "
            print(f"   #{doc['rank']} Score: {doc['relevance_score']:.4f} {cluster_tag}")
            print(f"      Title: {doc['title']} ({doc['id']})")
            print(f"      Cluster: {doc['cluster_name']}")
            print(f"      Excerpt: {doc['snippet']}")


def main():
    parser = argparse.ArgumentParser(description="Q1: Automatic Document Categorization CLI")
    parser.add_argument("--docs-dir", default="data/documents", help="Directory containing .txt documents")
    parser.add_argument("--method", choices=["kmeans", "hierarchical"], default="kmeans", help="Clustering method")
    parser.add_argument("--n-clusters", type=int, default=5, help="Number of clusters (default: 5)")
    parser.add_argument("--query", type=str, default=None, help="Query text to find relevant cluster and documents")
    parser.add_argument("--top-k", type=int, default=5, help="Number of relevant documents to display")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive query prompt")

    args = parser.parse_args()

    # Ensure dataset exists
    if not os.path.exists(args.docs_dir) or not os.listdir(args.docs_dir):
        print("Initializing sample dataset...")
        initialize_sample_dataset(".")

    docs = load_documents_from_directory(args.docs_dir)
    if not docs:
        print(f"Error: No text documents found in '{args.docs_dir}'.")
        sys.exit(1)

    print(f"Loaded {len(docs)} documents from '{args.docs_dir}'.")
    print(f"Running TF-IDF Vectorization and {args.method.upper()} Clustering (K={args.n_clusters})...")

    categorizer = DocumentCategorizer()
    categorizer.fit(docs)
    categorizer.cluster(method=args.method, n_clusters=args.n_clusters)

    groups = categorizer.get_clustered_groups()
    print_clusters(groups)

    if args.query:
        result = categorizer.query(args.query, top_k_docs=args.top_k)
        print_query_results(result)

    if args.interactive:
        print("\n" + format_separator("="))
        print("ENTERING INTERACTIVE QUERY MODE (Type 'exit' or 'quit' to end)")
        print(format_separator("="))
        while True:
            try:
                user_q = input("\nEnter query: ").strip()
                if user_q.lower() in ("exit", "quit", "q"):
                    print("Exiting interactive mode.")
                    break
                if not user_q:
                    continue
                res = categorizer.query(user_q, top_k_docs=args.top_k)
                print_query_results(res)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break


if __name__ == "__main__":
    main()
