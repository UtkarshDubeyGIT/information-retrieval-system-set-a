"""
Q1: Automatic Document Categorization System
============================================
This module converts text documents into numerical representations using TF-IDF,
clusters similar documents using K-Means or Hierarchical (Agglomerative) clustering,
displays documents grouped into clusters, and enables user query matching to identify
the most relevant cluster and documents with relevance scores.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class DocumentCategorizer:
    """
    Automatic Document Categorization System using TF-IDF and Clustering.
    """

    def __init__(
        self,
        stop_words: str = "english",
        max_df: float = 0.95,
        min_df: int = 1,
        ngram_range: Tuple[int, int] = (1, 2),
        sublinear_tf: bool = True,
    ):
        self.stop_words = stop_words
        self.max_df = max_df
        self.min_df = min_df
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf

        self.vectorizer = TfidfVectorizer(
            stop_words=self.stop_words,
            max_df=self.max_df,
            min_df=self.min_df,
            ngram_range=self.ngram_range,
            sublinear_tf=self.sublinear_tf,
            norm="l2",
        )

        self.documents: List[Dict[str, str]] = []
        self.tfidf_matrix: Optional[np.ndarray] = None
        self.feature_names: List[str] = []
        self.cluster_labels: Optional[np.ndarray] = None
        self.cluster_centroids: Optional[np.ndarray] = None
        self.cluster_names: Dict[int, str] = {}
        self.cluster_keywords: Dict[int, List[Tuple[str, float]]] = {}
        self.clustering_method: str = "kmeans"
        self.n_clusters: int = 3
        self.model = None

    def fit(self, documents: List[Dict[str, str]]) -> "DocumentCategorizer":
        """
        Fits the TF-IDF vectorizer on the collection of documents.
        Each document dict should contain 'content' (and optionally 'id', 'title', 'filename').
        """
        if not documents:
            raise ValueError("Document collection cannot be empty.")

        self.documents = documents
        corpus = [doc.get("content", "") for doc in documents]

        # Convert documents into numerical representations using TF-IDF
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus).toarray()
        self.feature_names = self.vectorizer.get_feature_names_out().tolist()
        return self

    def cluster(
        self,
        method: str = "kmeans",
        n_clusters: int = 5,
        random_state: int = 42,
        linkage: str = "ward",
    ) -> np.ndarray:
        """
        Groups similar documents using K-Means or Hierarchical (Agglomerative) clustering.
        """
        if self.tfidf_matrix is None:
            raise RuntimeError("Must call fit() before clustering.")

        n_samples = len(self.documents)
        n_clusters = min(n_clusters, n_samples)
        self.n_clusters = n_clusters
        self.clustering_method = method.lower()

        if self.clustering_method == "kmeans":
            self.model = KMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                n_init=10,
                max_iter=300,
            )
            self.cluster_labels = self.model.fit_predict(self.tfidf_matrix)
            self.cluster_centroids = self.model.cluster_centers_

        elif self.clustering_method in ("hierarchical", "agglomerative"):
            # Agglomerative hierarchical clustering
            # For ward linkage, Euclidean distance on L2-normalized TF-IDF is equivalent to cosine distance
            self.model = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=linkage,
            )
            self.cluster_labels = self.model.fit_predict(self.tfidf_matrix)

            # Compute centroids as the normalized mean of vectors in each cluster
            centroids = []
            for k in range(n_clusters):
                mask = self.cluster_labels == k
                if np.sum(mask) > 0:
                    mean_vec = np.mean(self.tfidf_matrix[mask], axis=0)
                    norm = np.linalg.norm(mean_vec)
                    if norm > 0:
                        mean_vec = mean_vec / norm
                    centroids.append(mean_vec)
                else:
                    centroids.append(np.zeros(self.tfidf_matrix.shape[1]))
            self.cluster_centroids = np.vstack(centroids)

        else:
            raise ValueError(f"Unsupported clustering method: {method}. Choose 'kmeans' or 'hierarchical'.")

        # Automatically extract descriptive keywords for each cluster
        self._generate_cluster_metadata()
        return self.cluster_labels

    def _generate_cluster_metadata(self, top_n_words: int = 6):
        """
        Extracts the top TF-IDF terms for each cluster centroid to generate human-readable labels.
        """
        self.cluster_keywords = {}
        self.cluster_names = {}

        for k in range(self.n_clusters):
            centroid = self.cluster_centroids[k]
            # Indices of top weights
            top_indices = np.argsort(centroid)[::-1][:top_n_words]
            top_terms = [
                (self.feature_names[idx], float(centroid[idx]))
                for idx in top_indices
                if centroid[idx] > 0
            ]
            self.cluster_keywords[k] = top_terms

            terms_str = ", ".join([term for term, _ in top_terms[:4]])
            self.cluster_names[k] = f"Cluster {k + 1}: {terms_str}" if terms_str else f"Cluster {k + 1}"

    def get_clustered_groups(self) -> List[Dict]:
        """
        Returns the documents grouped by cluster with metadata and centroid similarity.
        """
        if self.cluster_labels is None:
            raise RuntimeError("Must call cluster() before retrieving grouped documents.")

        groups = []
        for k in range(self.n_clusters):
            cluster_docs = []
            mask = np.where(self.cluster_labels == k)[0]

            for idx in mask:
                doc = self.documents[idx].copy()
                doc_vec = self.tfidf_matrix[idx : idx + 1]
                centroid_vec = self.cluster_centroids[k : k + 1]
                sim = float(cosine_similarity(doc_vec, centroid_vec)[0, 0])

                # Document excerpt
                content = doc.get("content", "")
                snippet = content[:150] + "..." if len(content) > 150 else content

                cluster_docs.append({
                    "index": int(idx),
                    "id": doc.get("id", f"doc_{idx}"),
                    "title": doc.get("title", f"Document {idx+1}"),
                    "filename": doc.get("filename", ""),
                    "snippet": snippet,
                    "similarity_to_centroid": round(sim, 4),
                    "content": content,
                })

            # Sort docs by similarity to centroid descending
            cluster_docs.sort(key=lambda x: x["similarity_to_centroid"], reverse=True)

            groups.append({
                "cluster_id": k,
                "cluster_number": k + 1,
                "cluster_name": self.cluster_names[k],
                "top_keywords": self.cluster_keywords[k],
                "doc_count": len(cluster_docs),
                "documents": cluster_docs,
            })

        return groups

    def query(self, query_text: str, top_k_docs: int = 5) -> Dict:
        """
        Given a user query:
        1. Vectorizes query into TF-IDF numerical space.
        2. Computes similarity to cluster centroids to identify the most relevant cluster.
        3. Computes similarity to all documents to identify and rank the most relevant documents.
        """
        if self.tfidf_matrix is None or self.cluster_centroids is None:
            raise RuntimeError("Must fit and cluster before querying.")

        query_clean = query_text.strip()
        if not query_clean:
            return {
                "query": query_text,
                "status": "empty_query",
                "best_cluster_id": -1,
                "best_cluster_name": "None",
                "best_cluster_score": 0.0,
                "cluster_scores": {},
                "ranked_documents": [],
            }

        # Vectorize query using fitted TF-IDF model
        query_vec = self.vectorizer.transform([query_clean]).toarray()
        query_norm = np.linalg.norm(query_vec)

        if query_norm == 0.0:
            # All words are out-of-vocabulary
            return {
                "query": query_text,
                "status": "out_of_vocabulary",
                "message": "Query terms were not found in the document vocabulary.",
                "best_cluster_id": 0,
                "best_cluster_name": self.cluster_names.get(0, "Cluster 1"),
                "best_cluster_score": 0.0,
                "cluster_scores": {k: 0.0 for k in range(self.n_clusters)},
                "ranked_documents": [],
            }

        # 1. Identify Most Relevant Cluster (cosine similarity against cluster centroids)
        cluster_sims = cosine_similarity(query_vec, self.cluster_centroids)[0]
        best_cluster_id = int(np.argmax(cluster_sims))
        best_cluster_score = float(cluster_sims[best_cluster_id])

        cluster_scores = {
            k: {
                "cluster_name": self.cluster_names[k],
                "score": round(float(cluster_sims[k]), 4),
            }
            for k in range(self.n_clusters)
        }

        # 2. Identify Most Relevant Documents (cosine similarity against document vectors)
        doc_sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        ranked_indices = np.argsort(doc_sims)[::-1]

        ranked_docs = []
        for rank, idx in enumerate(ranked_indices[:top_k_docs], start=1):
            score = float(doc_sims[idx])
            doc = self.documents[idx]
            content = doc.get("content", "")
            snippet = content[:180] + "..." if len(content) > 180 else content
            doc_cluster = int(self.cluster_labels[idx])

            ranked_docs.append({
                "rank": rank,
                "index": int(idx),
                "id": doc.get("id", f"doc_{idx}"),
                "title": doc.get("title", f"Document {idx+1}"),
                "cluster_id": doc_cluster,
                "cluster_name": self.cluster_names.get(doc_cluster, f"Cluster {doc_cluster+1}"),
                "is_in_best_cluster": bool(doc_cluster == best_cluster_id),
                "relevance_score": round(score, 4),
                "snippet": snippet,
            })

        return {
            "query": query_text,
            "status": "success",
            "best_cluster_id": best_cluster_id,
            "best_cluster_name": self.cluster_names[best_cluster_id],
            "best_cluster_score": round(best_cluster_score, 4),
            "cluster_scores": cluster_scores,
            "ranked_documents": ranked_docs,
        }

    def get_2d_projection(
        self, query_text: Optional[str] = None
    ) -> Dict[str, Union[np.ndarray, List]]:
        """
        Projects high-dimensional TF-IDF vectors to 2D for visualization using PCA.
        Includes document coordinates, centroid coordinates, and query coordinate if given.
        """
        if self.tfidf_matrix is None:
            raise RuntimeError("Must fit before projection.")

        # Fit PCA on document TF-IDF matrix
        pca = PCA(n_components=2, random_state=42)
        doc_coords_2d = pca.fit_transform(self.tfidf_matrix)

        # Project centroids
        if self.cluster_centroids is not None:
            centroid_coords_2d = pca.transform(self.cluster_centroids)
        else:
            centroid_coords_2d = np.empty((0, 2))

        # Project query if provided
        query_coord_2d = None
        if query_text:
            q_vec = self.vectorizer.transform([query_text]).toarray()
            if np.linalg.norm(q_vec) > 0:
                query_coord_2d = pca.transform(q_vec)[0]

        return {
            "doc_coords": doc_coords_2d,
            "centroid_coords": centroid_coords_2d,
            "query_coord": query_coord_2d,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cluster_labels": self.cluster_labels.tolist() if self.cluster_labels is not None else [],
            "titles": [d.get("title", "") for d in self.documents],
        }
