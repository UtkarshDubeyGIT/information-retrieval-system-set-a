#!/usr/bin/env python3
"""
CLI Application for Q2: Multimedia Search and Retrieval System.

Usage:
    python3 q2_cli.py --keyword "cardiovascular heart health"
    python3 q2_cli.py --image-query data/queries/query_green_leaf.png
    python3 q2_cli.py --keyword "ocean" --image-query data/queries/query_blue_water.png
    python3 q2_cli.py --list
    python3 q2_cli.py --interactive
"""

import argparse
import os
import sys
from src.q2_multimedia import MultimediaRetrievalSystem
from src.utils import initialize_sample_dataset


def format_separator(char="=", length=80):
    return char * length


def print_multimedia_results(results, query_desc=""):
    print("\n" + format_separator("="))
    print(f"MULTIMEDIA RETRIEVAL RESULTS {query_desc} ({len(results)} matches found)")
    print(format_separator("="))

    if not results:
        print("No matching records found.")
        return

    for rank, item in enumerate(results, start=1):
        media_icon = "📄 [TEXT]" if item.get("media_type") == "text" else "🖼️ [IMAGE]"
        score = item.get("relevance_score", 0.0)

        print(f"\n#{rank} {media_icon} {item.get('title')}  |  Relevance Score: {score:.4f}")
        print(f"    ID:          {item.get('id')}")
        print(f"    Category:    {item.get('category')}")
        print(f"    Upload Date: {item.get('upload_date')}")
        print(f"    File Path:   {item.get('file_path')}")
        if item.get("tags"):
            print(f"    Tags:        {', '.join(item['tags'])}")
        print(f"    Description: {item.get('description')}")
        if "matched_terms" in item:
            print(f"    Matched:     {', '.join(item['matched_terms'])}")
        if "text_score" in item and "visual_score" in item:
            print(f"    Fusion:      Text={item['text_score']:.4f}, Visual={item['visual_score']:.4f}")


def print_catalog_inventory(items):
    print("\n" + format_separator("="))
    print(f"MULTIMEDIA REPOSITORY INVENTORY ({len(items)} Items Registered)")
    print(format_separator("="))

    for item in items:
        badge = "[DOC]" if item.get("media_type") == "text" else "[IMG]"
        print(f"{badge:<6} {item.get('id'):<9} | {item.get('category'):<22} | {item.get('upload_date')} | {item.get('title')}")


def main():
    parser = argparse.ArgumentParser(description="Q2: Multimedia Search and Retrieval System CLI")
    parser.add_argument("--metadata-path", default="data/metadata.json", help="Path to metadata.json")
    parser.add_argument("--keyword", type=str, default=None, help="Keyword query string over metadata")
    parser.add_argument("--image-query", type=str, default=None, help="Path to query image file")
    parser.add_argument("--media-type", choices=["all", "text", "image"], default="all", help="Media type filter")
    parser.add_argument("--category", type=str, default=None, help="Category filter")
    parser.add_argument("--metric", choices=["intersection", "cosine", "bhattacharyya"], default="intersection", help="Color histogram metric")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--list", action="store_true", help="List all catalog items and metadata")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive search session")

    args = parser.parse_args()

    # Ensure dataset exists
    if not os.path.exists(args.metadata_path):
        print("Metadata repository not found. Initializing sample dataset...")
        initialize_sample_dataset(".")

    system = MultimediaRetrievalSystem(args.metadata_path, base_dir=".")

    if args.list:
        print_catalog_inventory(system.items)
        return

    # Check for direct queries
    if args.keyword or args.image_query:
        query_parts = []
        if args.keyword:
            query_parts.append(f"Keyword: '{args.keyword}'")
        if args.image_query:
            query_parts.append(f"Image: '{args.image_query}' ({args.metric})")
        desc = "for " + " + ".join(query_parts)

        results = system.unified_search(
            query_text=args.keyword,
            query_image_input=args.image_query,
            media_type=args.media_type,
            category_filter=args.category,
            top_k=args.top_k,
        )
        print_multimedia_results(results, desc)
        return

    if args.interactive:
        print("\n" + format_separator("="))
        print("ENTERING INTERACTIVE MULTIMEDIA SEARCH MODE (Type 'exit' to quit)")
        print(format_separator("="))
        while True:
            try:
                print("\nChoose search mode:")
                print("1. Keyword Metadata Search (documents & images)")
                print("2. Visual Image Search (Color Histogram)")
                print("3. Multimodal Search (Keyword + Image)")
                print("4. List All Items")
                print("5. Exit")
                choice = input("Enter choice (1-5): ").strip()

                if choice in ("5", "exit", "quit", "q"):
                    break
                elif choice == "1":
                    q = input("Enter search keywords: ").strip()
                    m_type = input("Filter by media type (all/text/image) [all]: ").strip() or "all"
                    results = system.search_metadata(q, media_type_filter=m_type, top_k=args.top_k)
                    print_multimedia_results(results, f"Keyword: '{q}'")
                elif choice == "2":
                    img_path = input("Enter image path (e.g., data/queries/query_green_leaf.png): ").strip()
                    if not os.path.exists(img_path):
                        print(f"Error: File '{img_path}' does not exist.")
                        continue
                    results = system.retrieve_similar_images(img_path, top_k=args.top_k, metric=args.metric)
                    print_multimedia_results(results, f"Image: '{img_path}'")
                elif choice == "3":
                    q = input("Enter keywords: ").strip()
                    img_path = input("Enter image path: ").strip()
                    results = system.unified_search(query_text=q, query_image_input=img_path, top_k=args.top_k)
                    print_multimedia_results(results, f"Fused Query: '{q}' + '{img_path}'")
                elif choice == "4":
                    print_catalog_inventory(system.items)
            except (KeyboardInterrupt, EOFError):
                break
    else:
        # Default run when no arguments provided: run standard showcase
        print("No query arguments provided. Running default sample search demonstrations:")
        print("\n1. Searching keyword 'space astronomy':")
        r1 = system.search_metadata("space astronomy", top_k=3)
        print_multimedia_results(r1, "Keyword: 'space astronomy'")

        print("\n2. Searching with visual color query 'data/queries/query_green_leaf.png':")
        r2 = system.retrieve_similar_images("data/queries/query_green_leaf.png", top_k=3)
        print_multimedia_results(r2, "Image Query: 'query_green_leaf.png'")


if __name__ == "__main__":
    main()
