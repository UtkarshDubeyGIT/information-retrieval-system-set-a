# Implementation Plan - Test Set A: Information Retrieval Systems

Develop an end-to-end, academic-grade and production-ready implementation for **Test - Set A** covering:
- **Q1. Automatic Document Categorization System**
- **Q2. Multimedia Search and Retrieval System**

Includes both an interactive modern **Streamlit Web Application**, high-fidelity **CLI/Standalone Scripts**, comprehensive modular engine (`src/`), a rich curated multimedia dataset, unit and integration test suite, and a submission-ready ZIP package.

---

## 1. System Architecture & Requirements Breakdown

### Q1. Automatic Document Categorization System
- **Numerical Representation**:
  - Preprocess text (tokenization, lowercase, punctuation removal, stopword filtering).
  - Compute TF-IDF feature matrices using scikit-learn's `TfidfVectorizer`.
  - Feature extraction with configurable n-gram ranges ($1, 2$) and sublinear TF scaling.
- **Clustering Techniques**:
  - **K-Means Clustering**: Partitioning into $K$ clusters with inertia evaluation and cluster centroid determination.
  - **Hierarchical Clustering (Agglomerative)**: Ward/Average linkage clustering, providing dendrogram or cluster assignments.
  - Automatic cluster labeling: Extract top TF-IDF keywords for each cluster to describe its semantic theme.
- **Cluster Visualization & Display**:
  - Group and display documents clearly under each cluster with top terms, document titles, and excerpt summaries.
  - 2D PCA / t-SNE projection scatter plot showing cluster separation in the Streamlit UI.
- **User Query & Relevance Matching**:
  - Accept user query string.
  - Vectorize query into the existing TF-IDF space.
  - Compute cosine similarity against cluster centroids to identify the **most relevant cluster**.
  - Compute cosine similarity against individual document vectors to rank and display the most relevant documents with exact match scores ($[0.0, 1.0]$).

---

### Q2. Multimedia Search and Retrieval System
- **Metadata Management**:
  - Store and manage metadata: `id`, `title`, `description`, `category`, `upload_date`, `media_type` (`text` / `image`), `file_path`, `tags`.
  - JSON and SQLite storage for persistent, indexed multimedia records.
- **Keyword-based Metadata Search**:
  - Inverted index / TF-IDF keyword search over metadata fields (`title`, `description`, `category`, `tags`).
  - Weighted relevance scoring (e.g., Title: 2.0x weight, Description: 1.0x, Category: 1.5x) normalized to $[0.0, 1.0]$.
  - Exact phrase and multi-keyword boolean query matching.
- **Image Retrieval using Color Histograms**:
  - Image feature extraction: 3D Color Histograms in RGB or HSV color space (8 bins per channel $\rightarrow$ 512-dimensional normalized feature vector).
  - Distance & similarity metrics:
    - Histogram Intersection Similarity ($\sum \min(H_1, H_2)$)
    - Cosine Similarity between normalized histograms
    - Chi-Square / Bhattacharyya distance converted to similarity score
  - Query-by-Example: Users can upload an image or select a catalog image to find visually similar images ranked by similarity score.
- **Result Presentation**:
  - Unified and filtered result cards showing metadata, thumbnails/text snippets, media type, and normalized relevance score.

---

## 2. Directory Layout

```
assignmentIrs/
├── data/
│   ├── documents/             # Sample text documents across diverse domains
│   ├── images/                # Sample images with distinct color profiles
│   └── metadata.json          # Pre-indexed metadata collection for Q2
├── src/
│   ├── __init__.py
│   ├── q1_categorization.py   # Document clustering, TF-IDF, Query matching
│   ├── q2_multimedia.py       # Metadata search, Histogram extraction, Image retrieval
│   └── utils.py               # Shared helpers, sample data generator
├── tests/
│   ├── __init__.py
│   ├── test_q1.py             # Unit tests for TF-IDF, K-means, Hierarchical, Querying
│   └── test_q2.py             # Unit tests for Metadata search, Color histograms, Image retrieval
├── app.py                     # Interactive Streamlit Web Application (Q1 & Q2)
├── q1_cli.py                  # Standalone CLI for Q1
├── q2_cli.py                  # Standalone CLI for Q2
├── run_demo.py                # End-to-end automated demo script
├── requirements.txt           # Project dependencies
├── README.md                  # Comprehensive documentation, setup guide, and theory
├── devlog/                    # Archived project devlogs
│   ├── phase1-implementation_plan.md
│   ├── phase1-task.md
│   └── phase1-walkthrough.md
└── assignment_irs_set_a.zip   # Final deliverable archive
```
