"""
Unit tests for Q2: Multimedia Search and Retrieval System.
"""

import os
import unittest
import numpy as np
from PIL import Image
from src.q2_multimedia import MultimediaRetrievalSystem
from src.utils import initialize_sample_dataset


class TestMultimediaRetrievalSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        _, _, cls.metadata_path = initialize_sample_dataset(".")
        cls.system = MultimediaRetrievalSystem(cls.metadata_path, base_dir=".")

    def test_metadata_loaded(self):
        self.assertGreater(len(self.system.items), 10)
        # Check required fields
        for item in self.system.items:
            self.assertIn("title", item)
            self.assertIn("description", item)
            self.assertIn("category", item)
            self.assertIn("upload_date", item)
            self.assertIn("media_type", item)
            self.assertIn("file_path", item)

    def test_keyword_search_documents(self):
        results = self.system.search_metadata("cardiovascular heart health", media_type_filter="text", top_k=5)
        self.assertGreater(len(results), 0)

        top_match = results[0]
        self.assertIn("cardio", top_match["title"].lower())
        self.assertGreater(top_match["relevance_score"], 0.4)
        self.assertIn("heart", [t.lower() for t in top_match["matched_terms"]])

    def test_keyword_search_images(self):
        results = self.system.search_metadata("forest green trees", media_type_filter="image", top_k=5)
        self.assertGreater(len(results), 0)

        top_match = results[0]
        self.assertIn("forest", top_match["title"].lower())
        self.assertEqual(top_match["media_type"], "image")

    def test_color_histogram_properties(self):
        sample_img_path = os.path.join("data", "images", "forest_lush_green.png")
        hist = self.system.extract_color_histogram(sample_img_path, bins_per_channel=16)

        # Check shape 16*3 = 48
        self.assertEqual(hist.shape, (48,))
        # Check normalized sum = 1.0
        self.assertAlmostEqual(float(hist.sum()), 1.0, places=5)
        # Check non-negative
        self.assertTrue(np.all(hist >= 0.0))

    def test_image_retrieval_exact_match(self):
        target_path = os.path.join("data", "images", "ocean_deep_blue.png")
        results = self.system.retrieve_similar_images(target_path, top_k=3, metric="intersection")

        self.assertGreater(len(results), 0)
        # The first match must be the exact same image
        self.assertIn("ocean_deep_blue", results[0]["file_path"])
        # Score should be ~ 1.0
        self.assertAlmostEqual(results[0]["relevance_score"], 1.0, places=3)

    def test_image_retrieval_color_discrimination(self):
        # Querying with a green query image should rank forest green highest
        green_query = os.path.join("data", "queries", "query_green_leaf.png")
        results = self.system.retrieve_similar_images(green_query, top_k=5, metric="intersection")

        top_match = results[0]
        self.assertIn("forest", top_match["file_path"].lower())

        # Check that green ranks much higher than sunset red or desert yellow
        red_score = next((r["relevance_score"] for r in results if "sunset" in r["file_path"]), 0.0)
        green_score = results[0]["relevance_score"]
        self.assertGreater(green_score, red_score)

    def test_different_metrics(self):
        img_path = os.path.join("data", "images", "desert_golden_yellow.png")
        for metric in ["intersection", "cosine", "bhattacharyya"]:
            results = self.system.retrieve_similar_images(img_path, top_k=3, metric=metric)
            self.assertGreater(len(results), 0)
            self.assertAlmostEqual(results[0]["relevance_score"], 1.0, places=2)

    def test_unified_search_multimodal(self):
        results = self.system.unified_search(
            query_text="ocean blue water",
            query_image_input=os.path.join("data", "queries", "query_blue_water.png"),
            top_k=5
        )
        self.assertGreater(len(results), 0)
        top = results[0]
        self.assertIn("ocean", top["title"].lower())
        self.assertTrue(0.0 <= top["relevance_score"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
