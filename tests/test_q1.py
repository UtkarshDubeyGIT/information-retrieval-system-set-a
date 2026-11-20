"""
Unit tests for Q1: Automatic Document Categorization System.
"""

import os
import unittest
import numpy as np
from src.q1_categorization import DocumentCategorizer
from src.utils import initialize_sample_dataset, load_documents_from_directory


class TestDocumentCategorizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        docs_dir, _, _ = initialize_sample_dataset(".")
        cls.docs = load_documents_from_directory(docs_dir)

    def test_fit_tfidf(self):
        categorizer = DocumentCategorizer()
        categorizer.fit(self.docs)

        self.assertIsNotNone(categorizer.tfidf_matrix)
        self.assertEqual(categorizer.tfidf_matrix.shape[0], len(self.docs))
        self.assertGreater(categorizer.tfidf_matrix.shape[1], 50)
        # Verify L2 normalization
        row_norms = np.linalg.norm(categorizer.tfidf_matrix, axis=1)
        for norm in row_norms:
            self.assertAlmostEqual(norm, 1.0, places=4)

    def test_kmeans_clustering(self):
        categorizer = DocumentCategorizer()
        categorizer.fit(self.docs)
        labels = categorizer.cluster(method="kmeans", n_clusters=5, random_state=42)

        self.assertEqual(len(labels), len(self.docs))
        self.assertEqual(len(np.unique(labels)), 5)
        self.assertEqual(categorizer.cluster_centroids.shape, (5, categorizer.tfidf_matrix.shape[1]))

    def test_hierarchical_clustering(self):
        categorizer = DocumentCategorizer()
        categorizer.fit(self.docs)
        labels = categorizer.cluster(method="hierarchical", n_clusters=5)

        self.assertEqual(len(labels), len(self.docs))
        self.assertEqual(len(np.unique(labels)), 5)
        self.assertEqual(categorizer.cluster_centroids.shape, (5, categorizer.tfidf_matrix.shape[1]))

    def test_get_clustered_groups(self):
        categorizer = DocumentCategorizer()
        categorizer.fit(self.docs)
        categorizer.cluster(method="kmeans", n_clusters=5)

        groups = categorizer.get_clustered_groups()
        self.assertEqual(len(groups), 5)

        total_docs = sum(g["doc_count"] for g in groups)
        self.assertEqual(total_docs, len(self.docs))

        for g in groups:
            self.assertIn("cluster_id", g)
            self.assertIn("cluster_name", g)
            self.assertIn("top_keywords", g)
            self.assertGreater(len(g["top_keywords"]), 0)
            for d in g["documents"]:
                self.assertIn("title", d)
                self.assertIn("similarity_to_centroid", d)
                self.assertTrue(0.0 <= d["similarity_to_centroid"] <= 1.0)

    def test_query_relevance_matching(self):
        categorizer = DocumentCategorizer()
        categorizer.fit(self.docs)
        categorizer.cluster(method="kmeans", n_clusters=5)

        # AI query
        query_text = "neural networks and deep learning models"
        result = categorizer.query(query_text, top_k_docs=5)

        self.assertEqual(result["status"], "success")
        self.assertGreaterEqual(result["best_cluster_id"], 0)
        self.assertGreater(result["best_cluster_score"], 0.0)

        ranked = result["ranked_documents"]
        self.assertGreater(len(ranked), 0)

        # Top ranked document should be AI related
        top_doc = ranked[0]
        self.assertIn("ai", top_doc["id"].lower())
        self.assertGreater(top_doc["relevance_score"], 0.3)
        self.assertTrue(0.0 <= top_doc["relevance_score"] <= 1.0)

    def test_space_query_relevance(self):
        categorizer = DocumentCategorizer()
        categorizer.fit(self.docs)
        categorizer.cluster(method="kmeans", n_clusters=5)

        query_text = "james webb telescope observing galaxies in space"
        result = categorizer.query(query_text, top_k_docs=5)

        top_doc = result["ranked_documents"][0]
        self.assertTrue(
            "astronomy" in top_doc["id"].lower() or "space" in top_doc["id"].lower() or "black_holes" in top_doc["id"].lower()
        )

    def test_out_of_vocabulary_query(self):
        categorizer = DocumentCategorizer()
        categorizer.fit(self.docs)
        categorizer.cluster(method="kmeans", n_clusters=5)

        result = categorizer.query("xyzqwerty nonexistentgibberishword", top_k_docs=5)
        self.assertEqual(result["status"], "out_of_vocabulary")

    def test_2d_projection(self):
        categorizer = DocumentCategorizer()
        categorizer.fit(self.docs)
        categorizer.cluster(method="kmeans", n_clusters=5)

        proj = categorizer.get_2d_projection(query_text="artificial intelligence")
        self.assertEqual(proj["doc_coords"].shape, (len(self.docs), 2))
        self.assertEqual(proj["centroid_coords"].shape, (5, 2))
        self.assertIsNotNone(proj["query_coord"])
        self.assertEqual(len(proj["query_coord"]), 2)


if __name__ == "__main__":
    unittest.main()
