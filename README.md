# Information Retrieval Systems — Test Set A

**Automatic Document Categorization & Multimedia Search and Retrieval System**

---

## 📌 Overview

This project delivers an end-to-end, academic-grade implementation for **Test - Set A**, solving two core problems in modern Information Retrieval:

1. **Q1. Automatic Document Categorization System**
   - Automatically groups a collection of text documents according to their content.
   - Converts documents into numerical representations using **TF-IDF**.
   - Groups similar documents using **K-Means** or **Hierarchical (Agglomerative)** clustering.
   - Displays documents grouped into different clusters with automatically derived thematic keywords and centroid similarity.
   - Allows a user to enter a query and identifies the **most relevant cluster** (via centroid cosine similarity) and **ranked documents** (with normalized relevance scores).

2. **Q2. Multimedia Search and Retrieval System**
   - Stores and retrieves text documents and images from a unified searchable collection.
   - Maintains structured metadata: `title`, `description`, `category`, `upload_date`, `media_type`, `file_path`, and `tags`.
   - Supports **keyword-based searching** of metadata with field-weighted relevance and TF-IDF cosine similarity.
   - Implements **content-based image retrieval (CBIR)** using **color histograms** with *Histogram Intersection*, *Cosine Similarity*, and *Bhattacharyya Distance*.
   - Displays matching documents and images with their complete metadata and calibrated relevance scores ($[0.0, 1.0]$).

The system includes both an interactive **Streamlit Web Application** (`app.py`), standalone **CLI interfaces** (`q1_cli.py`, `q2_cli.py`), an automated end-to-end demonstration runner (`run_demo.py`), and a complete unit test suite (`tests/`).

---

## 📁 Repository Structure

```
assignmentIrs/
├── app.py                     # Interactive Streamlit Web Application (Q1 & Q2)
├── q1_cli.py                  # Standalone CLI for Q1 Document Categorization
├── q2_cli.py                  # Standalone CLI for Q2 Multimedia Retrieval
├── run_demo.py                # Automated end-to-end demonstration runner
├── requirements.txt           # Project dependencies
├── README.md                  # Detailed documentation and system design
├── assignment_irs_set_a.zip   # Packaged submission archive
├── data/
│   ├── documents/             # 15 sample documents across 5 subject domains
│   │   ├── ai_deep_learning.txt
│   │   ├── nlp_transformers.txt
│   │   ├── cloud_microservices.txt
│   │   ├── cardiovascular_health.txt
│   │   ├── immunology_vaccines.txt
│   │   ├── neuroscience_brain.txt
│   │   ├── astronomy_telescopes.txt
│   │   ├── mars_planetary_science.txt
│   │   ├── black_holes_gravity.txt
│   │   ├── macroeconomics_inflation.txt
│   │   ├── stock_market_investing.txt
│   │   ├── crypto_fintech.txt
│   │   ├── football_tactics.txt
│   │   ├── tennis_grand_slam.txt
│   │   └── marathon_endurance.txt
│   ├── images/                # Catalog images with distinct color profiles
│   │   ├── forest_lush_green.png
│   │   ├── ocean_deep_blue.png
│   │   ├── sunset_crimson_red.png
│   │   ├── desert_golden_yellow.png
│   │   └── galaxy_neon_purple.png
│   ├── queries/               # Sample query images for color histogram testing
│   │   ├── query_green_leaf.png
│   │   ├── query_blue_water.png
│   │   └── query_red_fire.png
│   └── metadata.json          # Indexed multimedia repository metadata
├── src/
│   ├── __init__.py
│   ├── q1_categorization.py   # TF-IDF, K-Means, Hierarchical Clustering, Query Matching
│   ├── q2_multimedia.py       # Metadata Search, Color Histogram Extraction, Image Retrieval
│   └── utils.py               # Dataset generation and indexing utilities
├── tests/
│   ├── __init__.py
│   ├── test_q1.py             # 8 unit tests for Q1
│   └── test_q2.py             # 8 unit tests for Q2
└── devlog/                    # Archived implementation devlogs
    ├── phase1-implementation_plan.md
    ├── phase1-task.md
    └── phase1-walkthrough.md
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites & Setup
Ensure Python 3.9+ is installed. Install the required dependencies:

```bash
pip install -r requirements.txt
```

*(Dependencies: `numpy`, `scikit-learn`, `scipy`, `pillow`, `matplotlib`, `streamlit`)*

---

### 2. Launch the Streamlit Web Application
Run the modern interactive web interface:

```bash
streamlit run app.py
```

The application provides:
- **Tab 1: Q1 Document Categorization**
  - Interactive clustering method switch (`K-Means` vs `Hierarchical`).
  - Slider to change $K$ dynamically.
  - Expandable cluster cards showing top TF-IDF keywords, document counts, and member summaries.
  - Live query input identifying the **most relevant cluster** with a cluster similarity distribution chart.
  - Ranked documents table with relevance scores $\in [0.0, 1.0]$.
  - 2D PCA semantic scatter plot showing clusters, centroids, and the query point.
- **Tab 2: Q2 Multimedia Search & Retrieval**
  - Keyword search on document/image metadata.
  - Image retrieval using color histograms (upload an image or test preset images).
  - Multimodal fusion search combining text keywords and visual features.
  - Metadata filters (Media Type, Category).
  - Color histogram spectrum visualizer comparing channel distributions.
- **Tab 3: Architectural & Theoretical Notes**
  - Mathematical equations and methodology documentation.

---

### 3. Run Standalone CLI Tools

#### Q1: Document Categorization CLI (`q1_cli.py`)
```bash
# Query the collection with K-Means
python3 q1_cli.py --query "artificial intelligence deep learning neural networks"

# Use Hierarchical clustering with K=5
python3 q1_cli.py --method hierarchical --n-clusters 5 --query "cardiovascular heart health"

# Launch interactive query prompt
python3 q1_cli.py --interactive
```

#### Q2: Multimedia Search CLI (`q2_cli.py`)
```bash
# Keyword-based metadata search
python3 q2_cli.py --keyword "cardiovascular heart disease"

# Color histogram image retrieval
python3 q2_cli.py --image-query data/queries/query_green_leaf.png

# Multimodal combined search (Text + Image)
python3 q2_cli.py --keyword "deep ocean" --image-query data/queries/query_blue_water.png

# List all catalog items and metadata
python3 q2_cli.py --list

# Interactive multimedia search prompt
python3 q2_cli.py --interactive
```

---

### 4. Run Automated Demonstration
Execute both Q1 and Q2 end-to-end demonstrations with a single command:

```bash
python3 run_demo.py
```

---

### 5. Run the Automated Unit Test Suite
Verify all 16 tests covering TF-IDF, clustering, query relevance, metadata search, color histograms, and image retrieval:

```bash
python3 -m unittest discover -s tests -v
```

---

## 🧠 Algorithmic and Mathematical Foundations

### Q1: Automatic Document Categorization
1. **TF-IDF Numerical Representation**:
   $$\text{TF-IDF}(t, d, D) = \left(1 + \log(\text{tf}(t, d))\right) \times \left(1 + \log\left(\frac{1 + |D|}{1 + \text{df}(t)}\right)\right)$$
   Each document vector $\mathbf{v}_d$ is $L_2$-normalized: $\hat{\mathbf{v}}_d = \frac{\mathbf{v}_d}{\|\mathbf{v}_d\|_2}$.

2. **Clustering**:
   - **K-Means Clustering**: Minimizes the within-cluster inertia:
     $$J = \sum_{k=1}^K \sum_{d \in C_k} \|\hat{\mathbf{v}}_d - \boldsymbol{\mu}_k\|^2$$
   - **Hierarchical Agglomerative Clustering**: Uses Ward's minimum variance criterion to sequentially merge document clusters.
   - **Automatic Cluster Naming**: Extracts the top $M$ terms with the highest centroid weights.

3. **Query Relevance & Matching**:
   Given a user query string $q$, it is mapped into the TF-IDF space as $\mathbf{q}$.
   - **Most Relevant Cluster**:
     $$k^* = \arg\max_k \cos(\mathbf{q}, \boldsymbol{\mu}_k) = \frac{\mathbf{q} \cdot \boldsymbol{\mu}_k}{\|\mathbf{q}\|_2 \|\boldsymbol{\mu}_k\|_2}$$
   - **Ranked Relevant Documents**:
     $$\text{Relevance}(d) = \cos(\mathbf{q}, \hat{\mathbf{v}}_d) \in [0.0, 1.0]$$

---

### Q2: Multimedia Search and Retrieval System
1. **Metadata Management & Keyword Search**:
   Maintains title, description, category, upload date, and tags. Searches calculate a composite score combining field-weighted term matching:
   - Title: $3.0\times$
   - Category: $2.0\times$
   - Tags: $2.5\times$
   - Description: $1.0\times$
   Blended with TF-IDF cosine similarity over metadata descriptions.

2. **Image Retrieval via Color Histograms**:
   - Extracts a normalized color histogram in RGB color space with 16 bins per channel ($16 \times 3 = 48$ bins).
   - $L_1$-normalized: $\sum_{i=1}^{48} H[i] = 1.0$.
   - **Histogram Intersection (Swain & Ballard)**:
     $$K(H_q, H_t) = \sum_{i=1}^{48} \min(H_q[i], H_t[i])$$
     Provides a bounded similarity metric in $[0.0, 1.0]$.
   - Also supports **Cosine Similarity** and **Bhattacharyya Coefficient**.
