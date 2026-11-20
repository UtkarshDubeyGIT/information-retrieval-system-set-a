"""
Q2: Multimedia Search and Retrieval System
==========================================
Maintains metadata (title, description, category, upload date), supports
keyword-based searching of document metadata, and implements image retrieval
using 3D color histograms with histogram intersection and cosine similarity metrics.
Displays matching documents and images with metadata and relevance scores.
"""

import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MultimediaRetrievalSystem:
    """
    Multimedia Search and Retrieval Engine for documents and images.
    """

    def __init__(self, metadata_path: Optional[str] = None, base_dir: str = "."):
        self.base_dir = base_dir
        self.items: List[Dict] = []
        self.image_histograms: Dict[str, np.ndarray] = {}
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.metadata_tfidf_matrix: Optional[np.ndarray] = None
        self.metadata_path = metadata_path

        if metadata_path and os.path.exists(metadata_path):
            self.load_metadata(metadata_path)

    def load_metadata(self, metadata_path: str):
        """Loads metadata repository and precomputes feature indices."""
        self.metadata_path = metadata_path
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.items = json.load(f)

        self._build_metadata_index()
        self._build_image_histogram_index()

    def add_item(
        self,
        item_id: str,
        title: str,
        description: str,
        category: str,
        upload_date: Optional[str] = None,
        media_type: str = "text",
        file_path: str = "",
        tags: Optional[List[str]] = None,
    ):
        """Adds a new multimedia item into the collection."""
        if not upload_date:
            upload_date = datetime.now().strftime("%Y-%m-%d")

        item = {
            "id": item_id,
            "title": title,
            "description": description,
            "category": category,
            "upload_date": upload_date,
            "media_type": media_type.lower(),
            "file_path": file_path,
            "tags": tags or [],
        }
        self.items.append(item)

        # Update indices
        self._build_metadata_index()
        if item["media_type"] == "image":
            full_path = os.path.join(self.base_dir, file_path) if not os.path.isabs(file_path) else file_path
            if os.path.exists(full_path):
                self.image_histograms[item_id] = self.extract_color_histogram(full_path)

    def _build_metadata_index(self):
        """Builds TF-IDF index over metadata fields for keyword retrieval."""
        if not self.items:
            return

        corpus = []
        for item in self.items:
            # Emphasize title, category and tags in the searchable corpus
            tags_str = " ".join(item.get("tags", []))
            text = (
                f"{item.get('title', '')} {item.get('title', '')} "
                f"{item.get('category', '')} {tags_str} {tags_str} "
                f"{item.get('description', '')}"
            )
            corpus.append(text)

        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words="english",
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.metadata_tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus).toarray()

    def _build_image_histogram_index(self):
        """Computes and caches 3D color histograms for all image items."""
        self.image_histograms = {}
        for item in self.items:
            if item.get("media_type") == "image":
                fpath = item.get("file_path", "")
                full_path = os.path.join(self.base_dir, fpath) if not os.path.isabs(fpath) else fpath
                if os.path.exists(full_path):
                    hist = self.extract_color_histogram(full_path)
                    self.image_histograms[item["id"]] = hist

    @staticmethod
    def extract_color_histogram(
        image_input: Union[str, Image.Image],
        bins_per_channel: int = 16,
    ) -> np.ndarray:
        """
        Extracts a normalized color histogram in RGB color space.
        Computes 16 bins for Red, Green, and Blue channels (48-dimensional feature vector).
        Normalized using L1 norm so that sum(H) == 1.0.
        """
        if isinstance(image_input, str):
            img = Image.open(image_input).convert("RGB")
        else:
            img = image_input.convert("RGB")

        # Resize for consistent and rapid feature extraction
        img = img.resize((128, 128), Image.Resampling.BILINEAR)
        arr = np.array(img, dtype=float)

        r_hist, _ = np.histogram(arr[:, :, 0], bins=bins_per_channel, range=(0, 256))
        g_hist, _ = np.histogram(arr[:, :, 1], bins=bins_per_channel, range=(0, 256))
        b_hist, _ = np.histogram(arr[:, :, 2], bins=bins_per_channel, range=(0, 256))

        # Concatenate channel histograms into a single feature vector
        hist = np.concatenate([r_hist, g_hist, b_hist]).astype(float)
        total = float(hist.sum())
        if total > 0:
            hist /= total

        return hist

    def search_metadata(
        self,
        query: str,
        category_filter: Optional[str] = None,
        media_type_filter: str = "all",
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Keyword-based search over document metadata (title, description, category, tags).
        Computes a composite relevance score combining exact keyword matches and TF-IDF cosine similarity.
        """
        clean_query = query.strip()
        if not clean_query or not self.items:
            return []

        tokens = [t.lower() for t in re.findall(r"\w+", clean_query)]
        if not tokens:
            return []

        # 1. TF-IDF Cosine Similarity over metadata corpus
        tfidf_scores = np.zeros(len(self.items))
        if self.tfidf_vectorizer is not None and self.metadata_tfidf_matrix is not None:
            q_vec = self.tfidf_vectorizer.transform([clean_query]).toarray()
            if np.linalg.norm(q_vec) > 0:
                tfidf_scores = cosine_similarity(q_vec, self.metadata_tfidf_matrix)[0]

        results = []
        for idx, item in enumerate(self.items):
            # Apply filters
            if media_type_filter != "all" and item.get("media_type") != media_type_filter:
                continue
            if category_filter and item.get("category", "").lower() != category_filter.lower():
                continue

            # Field-weighted keyword matching
            title_text = item.get("title", "").lower()
            desc_text = item.get("description", "").lower()
            cat_text = item.get("category", "").lower()
            tags_text = " ".join(item.get("tags", [])).lower()

            match_score = 0.0
            matched_terms = set()

            for token in tokens:
                matched = False
                if token in title_text:
                    match_score += 3.0
                    matched = True
                if token in tags_text:
                    match_score += 2.5
                    matched = True
                if token in cat_text:
                    match_score += 2.0
                    matched = True
                if token in desc_text:
                    match_score += 1.0
                    matched = True
                if matched:
                    matched_terms.add(token)

            # Keyword coverage bonus (fraction of query terms present)
            coverage = len(matched_terms) / float(len(tokens))
            normalized_kw_score = min(1.0, (match_score / (len(tokens) * 3.5)) * 0.7 + coverage * 0.3)
            tfidf_sim = float(tfidf_scores[idx])

            # Blended relevance score: 60% TF-IDF cosine similarity + 40% keyword match
            composite_score = round(0.6 * tfidf_sim + 0.4 * normalized_kw_score, 4)

            # Include if there is any lexical match or TF-IDF relevance
            if composite_score > 0.01 or matched_terms:
                res_item = item.copy()
                res_item["relevance_score"] = max(composite_score, 0.05 if matched_terms else 0.0)
                res_item["matched_terms"] = list(matched_terms)
                results.append(res_item)

        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:top_k]

    def retrieve_similar_images(
        self,
        query_image_input: Union[str, Image.Image],
        top_k: int = 5,
        metric: str = "intersection",
    ) -> List[Dict]:
        """
        Retrieves matching images using 3D color histogram comparison.
        Supported metrics:
          - 'intersection': Histogram intersection sum(min(H1, H2)) -> [0.0, 1.0]
          - 'cosine': Cosine similarity dot(H1, H2) / (||H1|| * ||H2||) -> [0.0, 1.0]
          - 'bhattacharyya': 1.0 - sqrt(1 - sum(sqrt(H1 * H2))) -> [0.0, 1.0]
        """
        query_hist = self.extract_color_histogram(query_image_input)
        scored_images = []

        image_items = [it for it in self.items if it.get("media_type") == "image"]

        for item in image_items:
            item_id = item["id"]
            if item_id in self.image_histograms:
                hist = self.image_histograms[item_id]
            else:
                fpath = item.get("file_path", "")
                full_path = os.path.join(self.base_dir, fpath) if not os.path.isabs(fpath) else fpath
                if not os.path.exists(full_path):
                    continue
                hist = self.extract_color_histogram(full_path)
                self.image_histograms[item_id] = hist

            if metric == "intersection":
                # Histogram Intersection (Swain & Ballard)
                score = float(np.sum(np.minimum(query_hist, hist)))
            elif metric == "cosine":
                denom = np.linalg.norm(query_hist) * np.linalg.norm(hist)
                score = float(np.dot(query_hist, hist) / (denom + 1e-9))
            elif metric == "bhattacharyya":
                bc_coeff = float(np.sum(np.sqrt(query_hist * hist)))
                score = float(np.clip(bc_coeff, 0.0, 1.0))
            else:
                raise ValueError(f"Unknown metric '{metric}'. Use 'intersection', 'cosine', or 'bhattacharyya'.")

            res_item = item.copy()
            res_item["relevance_score"] = round(float(np.clip(score, 0.0, 1.0)), 4)
            res_item["metric_used"] = metric
            scored_images.append(res_item)

        scored_images.sort(key=lambda x: x["relevance_score"], reverse=True)
        return scored_images[:top_k]

    def unified_search(
        self,
        query_text: Optional[str] = None,
        query_image_input: Optional[Union[str, Image.Image]] = None,
        media_type: str = "all",
        category_filter: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Unified search handling text metadata search, visual color histogram search,
        or multimodal retrieval combining both with calibrated relevance scores.
        """
        has_text = bool(query_text and query_text.strip())
        has_image = bool(query_image_input is not None)

        if not has_text and not has_image:
            # Return collection listing up to top_k
            filtered = [
                it.copy() for it in self.items
                if (media_type == "all" or it.get("media_type") == media_type)
                and (not category_filter or it.get("category") == category_filter)
            ]
            for it in filtered:
                it["relevance_score"] = 1.0
            return filtered[:top_k]

        if has_text and not has_image:
            return self.search_metadata(
                query=query_text,
                category_filter=category_filter,
                media_type_filter=media_type,
                top_k=top_k,
            )

        if has_image and not has_text:
            return self.retrieve_similar_images(
                query_image_input=query_image_input,
                top_k=top_k,
                metric="intersection",
            )

        # Multimodal fusion: both text query and query image provided
        text_results = {
            it["id"]: it["relevance_score"]
            for it in self.search_metadata(query=query_text, media_type_filter="all", top_k=len(self.items))
        }
        image_results = {
            it["id"]: it["relevance_score"]
            for it in self.retrieve_similar_images(query_image_input=query_image_input, top_k=len(self.items))
        }

        fused_items = []
        for item in self.items:
            if media_type != "all" and item.get("media_type") != media_type:
                continue
            if category_filter and item.get("category") != category_filter:
                continue

            item_id = item["id"]
            t_score = text_results.get(item_id, 0.0)
            img_score = image_results.get(item_id, 0.0)

            if item["media_type"] == "image":
                fused_score = 0.5 * t_score + 0.5 * img_score
            else:
                fused_score = t_score

            if fused_score > 0.02:
                res = item.copy()
                res["relevance_score"] = round(fused_score, 4)
                res["text_score"] = round(t_score, 4)
                res["visual_score"] = round(img_score, 4) if item["media_type"] == "image" else 0.0
                fused_items.append(res)

        fused_items.sort(key=lambda x: x["relevance_score"], reverse=True)
        return fused_items[:top_k]
