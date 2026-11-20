# Academic Project Brief & Submission Report

# Information Retrieval Systems — Test Set A
**Comprehensive System Design, Theoretical Formulation, Implementation & Empirical Evaluation**

---

## Metadata & Submission Details

- **Assignment Title:** Information Retrieval Systems (IRS) — Test Set A
- **Deliverables:**
  1. **Source Codebase Archive:** `assignment_irs_set_a.zip`
  2. **GitHub Repository:** [https://github.com/UtkarshDubeyGIT/information-retrieval-system-set-a](https://github.com/UtkarshDubeyGIT/information-retrieval-system-set-a)
  3. **Interactive Demonstration:** Streamlit Web Application (`app.py`), CLIs (`q1_cli.py`, `q2_cli.py`), Demo Runner (`run_demo.py`)
- **Technology Stack:** Python 3.9+, Scikit-Learn, NumPy, SciPy, Pillow, Matplotlib, Streamlit

---

## 1. Executive Summary

This report presents a complete, production-grade implementation of two core Information Retrieval challenges specified in **Test - Set A**:

- **Question 1 (Document Categorization):** An automatic text categorization engine that processes unstructured documents into high-dimensional TF-IDF feature vectors, performs unsupervised grouping via K-Means and Hierarchical Agglomerative clustering, extracts descriptive thematic cluster labels, and resolves user queries by scoring cosine similarity against cluster centroids and document representations.
- **Question 2 (Multimedia Search & Retrieval):** A multimodal storage and retrieval system indexing text documents and images. It provides multi-field keyword retrieval over metadata with coverage and TF-IDF blending, content-based image retrieval (CBIR) using 48-dimensional RGB color histograms with Histogram Intersection, and unified multimodal query fusion.

---

## 2. Q1 — Automatic Document Categorization System

### 2.1 Problem Formulation & Objectives
The system automatically groups a corpus of text documents according to topical content without manual supervision. The operational workflow consists of:
1. Converting raw text documents into numerical vector representations.
2. Grouping documents into coherent clusters using unsupervised learning.
3. Automatically generating human-readable cluster topic descriptors.
4. Enabling user query submission to determine the **most relevant cluster** and retrieve a ranked list of relevant documents with quantitative scores.

### 2.2 Numerical Vectorization: TF-IDF Model
Documents are mapped into the Vector Space Model (VSM) using Term Frequency-Inverse Document Frequency (TF-IDF):
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

Where:
- **Sublinear Term Frequency:**
  $$\text{TF}(t, d) = \begin{cases} 1 + \log(\text{tf}(t, d)) & \text{if } \text{tf}(t, d) > 0 \\ 0 & \text{otherwise} \end{cases}$$
- **Smoothed Inverse Document Frequency:**
  $$\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + \text{df}(t, D)}\right) + 1$$
- **$L_2$ Normalization:** Each document vector $\mathbf{v}_d$ is projected onto the unit hypersphere:
  $$\hat{\mathbf{v}}_d = \frac{\mathbf{v}_d}{\|\mathbf{v}_d\|_2}$$

### 2.3 Clustering Techniques
The system supports two clustering paradigms:
1. **K-Means Clustering:**
   Partitions the normalized document vectors into $K$ disjoint clusters by minimizing within-cluster inertia:
   $$J = \sum_{k=1}^K \sum_{d \in C_k} \|\hat{\mathbf{v}}_d - \boldsymbol{\mu}_k\|^2$$
   The cluster centroid $\boldsymbol{\mu}_k = \frac{1}{|C_k|} \sum_{d \in C_k} \hat{\mathbf{v}}_d$ represents the semantic center of mass.
2. **Hierarchical Agglomerative Clustering:**
   Constructs a hierarchical cluster tree using **Ward's Linkage**, which iteratively merges clusters to minimize the increase in total within-cluster variance. Centroids are subsequently computed as normalized cluster means.

### 2.4 Automatic Cluster Labeling
For each cluster $k$, the top $M$ terms with the highest weights in centroid vector $\boldsymbol{\mu}_k$ are identified:
$$\text{Keywords}(C_k) = \operatorname{argtop}_{t} (\mu_{k, t})$$
These terms are automatically formatted into readable cluster themes (e.g., *Cluster 4: neural, memory, billions, learning*).

### 2.5 Query-to-Cluster & Query-to-Document Relevance Matching
Given a user query string $q$:
1. $q$ is transformed into vector $\mathbf{q}$ using the fitted TF-IDF vocabulary.
2. **Most Relevant Cluster Identification:**
   $$k^* = \arg\max_{k \in \{1,\dots,K\}} \cos(\mathbf{q}, \boldsymbol{\mu}_k) = \frac{\mathbf{q} \cdot \boldsymbol{\mu}_k}{\|\mathbf{q}\|_2 \|\boldsymbol{\mu}_k\|_2}$$
3. **Document Ranking:**
   $$\text{Score}(d) = \cos(\mathbf{q}, \hat{\mathbf{v}}_d) = \frac{\mathbf{q} \cdot \hat{\mathbf{v}}_d}{\|\mathbf{q}\|_2} \in [0.0, 1.0]$$
   Documents are sorted in descending order of relevance score.

### 2.6 Empirical Validation Results
| Query Input | Identified Best Cluster | Top Retrieved Document | Relevance Score |
| :--- | :--- | :--- | :---: |
| `"artificial intelligence neural networks"` | Cluster 4 (neural, learning) | `Ai Deep Learning` | **0.3206** |
| `"cardiovascular disease cholesterol"` | Cluster 1 (heart, health) | `Cardiovascular Health` | **0.2882** |
| `"james webb telescope deep space"` | Cluster 3 (astronomy, galaxies) | `Astronomy Telescopes` | **0.3415** |

---

## 3. Q2 — Multimedia Search and Retrieval System

### 3.1 Metadata Architecture
Every multimedia asset in the collection is modeled with structured metadata stored in `data/metadata.json`:
- `id`: Unique identifier (e.g., `DOC-001`, `IMG-001`)
- `title`: Descriptive title
- `description`: Textual summary / synopsis
- `category`: Domain category
- `upload_date`: Ingestion date (YYYY-MM-DD)
- `media_type`: `"text"` or `"image"`
- `file_path`: Relative filesystem path
- `tags`: List of descriptive keywords

### 3.2 Keyword-Based Metadata Search
The search engine parses multi-word queries and evaluates metadata fields with hierarchical weighting:
- **Title Weight:** $3.0\times$
- **Tags Weight:** $2.5\times$
- **Category Weight:** $2.0\times$
- **Description Weight:** $1.0\times$

The final metadata relevance score blends lexical coverage with TF-IDF cosine similarity across metadata fields:
$$\text{Score}_{\text{text}} = 0.6 \cdot \text{Sim}_{\text{TF-IDF}} + 0.4 \cdot \text{Score}_{\text{Keyword}}$$

### 3.3 Content-Based Image Retrieval (CBIR) via Color Histograms
1. **Feature Extraction:**
   - Color images are analyzed in the RGB color space.
   - For each channel ($R, G, B$), pixel intensities $[0, 255]$ are discretized into $B = 16$ bins.
   - Channel histograms are concatenated into a **48-dimensional feature vector** $\mathbf{H}$.
   - The vector is $L_1$-normalized: $\sum_{i=1}^{48} H[i] = 1.0$.
2. **Similarity Metrics:**
   - **Histogram Intersection (Swain & Ballard):**
     $$K(\mathbf{H}_q, \mathbf{H}_t) = \sum_{i=1}^{48} \min(H_q[i], H_t[i]) \in [0.0, 1.0]$$
   - **Cosine Similarity:**
     $$\cos(\mathbf{H}_q, \mathbf{H}_t) = \frac{\mathbf{H}_q \cdot \mathbf{H}_t}{\|\mathbf{H}_q\|_2 \|\mathbf{H}_t\|_2}$$
   - **Bhattacharyya Coefficient:**
     $$\text{BC}(\mathbf{H}_q, \mathbf{H}_t) = \sum_{i=1}^{48} \sqrt{H_q[i] \cdot H_t[i]}$$

### 3.4 Unified Multimodal Search
When a user provides both text keywords and a query image, the system fuses both modalities:
$$\text{Score}_{\text{multimodal}} = 0.5 \cdot \text{Score}_{\text{text}} + 0.5 \cdot \text{Score}_{\text{visual}}$$

### 3.5 Empirical Color Discrimination Results
Querying with an independent green botanical sample (`query_green_leaf.png`) against the catalog:
- `forest_lush_green.png`: Similarity **0.3172** (Top Match)
- `ocean_deep_blue.png`: Similarity **0.0465**
- `sunset_crimson_red.png`: Similarity **0.0000**

---

## 4. Software Architecture & Interfaces

```
assignmentIrs/
├── app.py                     # Interactive Streamlit Web Application
├── q1_cli.py                  # Standalone CLI for Q1
├── q2_cli.py                  # Standalone CLI for Q2
├── run_demo.py                # Automated end-to-end verification demo
├── requirements.txt           # Python dependency specification
├── README.md                  # System manual and instructions
├── SUBMISSION_BRIEF.md        # Academic submission report
├── doc_brief.html             # Styled HTML printable brief
├── data/
│   ├── documents/             # 15 domain-diverse text documents
│   ├── images/                # 5 distinct color profile catalog images
│   ├── queries/               # 3 independent evaluation query images
│   └── metadata.json          # Pre-indexed metadata repository
├── src/
│   ├── q1_categorization.py   # TF-IDF, K-Means, Hierarchical, Query matching
│   ├── q2_multimedia.py       # Metadata indexing, Color histograms, CBIR
│   └── utils.py               # Dataset generation and indexing utilities
├── tests/
│   ├── test_q1.py             # 8 unit tests for Q1
│   └── test_q2.py             # 8 unit tests for Q2
└── devlog/                    # Archived implementation plans and logs
```

---

## 5. Verification & Quality Assurance

The implementation was validated using automated test suites and end-to-end script executions:
- **Unit Tests:** 16 tests executed across `test_q1.py` and `test_q2.py`:
  - TF-IDF matrix generation and $L_2$ normalization verification: **PASSED**
  - K-Means cluster partitioning and centroid convergence: **PASSED**
  - Hierarchical agglomerative clustering assignment: **PASSED**
  - Cluster descriptor generation and keyword extraction: **PASSED**
  - Query vectorization and cluster identification: **PASSED**
  - Out-of-vocabulary query handling: **PASSED**
  - PCA 2D semantic space projection: **PASSED**
  - Metadata record parsing and indexing: **PASSED**
  - Weighted keyword metadata search: **PASSED**
  - 48-bin color histogram normalization ($\sum H = 1.0$): **PASSED**
  - Exact image retrieval match ($\text{score} \approx 1.0$): **PASSED**
  - Visual color discrimination across distinct color spectra: **PASSED**
  - Histogram Intersection, Cosine, and Bhattacharyya metrics: **PASSED**
  - Multimodal query score fusion: **PASSED**

---

## 6. How to Run the System

### 6.1 Interactive Web Application
```bash
streamlit run app.py
```
Provides visual cluster cards, interactive query evaluation, PCA 2D scatter plots, color histogram distribution bar charts, and image previews.

### 6.2 Standalone CLIs
```bash
# Q1: Document Categorization
python3 q1_cli.py --query "artificial intelligence neural networks" --n-clusters 5

# Q2: Multimedia Retrieval
python3 q2_cli.py --keyword "cardiovascular" --image-query data/queries/query_green_leaf.png
```

### 6.3 Automated Demonstration
```bash
python3 run_demo.py
```

### 6.4 Unit Test Suite
```bash
python3 -m unittest discover -s tests -v
```

---

## 7. Submission Artifacts Summary

| Artifact | Location | Description |
| :--- | :--- | :--- |
| **Codebase Archive** | `assignment_irs_set_a.zip` | Full self-contained archive ready for extraction and execution |
| **GitHub Repository** | [UtkarshDubeyGIT/information-retrieval-system-set-a](https://github.com/UtkarshDubeyGIT/information-retrieval-system-set-a) | Public git repository containing source code and history |
| **Documentation Brief** | `SUBMISSION_BRIEF.md` / `doc_brief.html` | Comprehensive academic project brief and implementation report |
