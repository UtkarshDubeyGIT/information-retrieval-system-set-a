"""
Information Retrieval Systems - Test Set A
Interactive Web Application powered by Streamlit.
Covers:
  - Q1: Automatic Document Categorization System (TF-IDF + K-Means / Hierarchical + Query Matching)
  - Q2: Multimedia Search and Retrieval System (Metadata Search + Color Histograms + Unified Retrieval)
"""

import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from src.q1_categorization import DocumentCategorizer
from src.q2_multimedia import MultimediaRetrievalSystem
from src.utils import (
    initialize_sample_dataset,
    load_documents_from_directory,
    load_metadata_store,
)

# Page Configuration
st.set_page_config(
    page_title="IRS Test - Set A | Document & Multimedia Retrieval",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize sample dataset if needed
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
docs_dir, images_dir, metadata_path = initialize_sample_dataset(BASE_DIR)


@st.cache_resource
def get_systems(method="kmeans", n_clusters=5, ngram_max=2):
    """Caches the document categorizer and multimedia retrieval systems."""
    docs = load_documents_from_directory(os.path.join(BASE_DIR, "data", "documents"))
    categorizer = DocumentCategorizer(ngram_range=(1, ngram_max))
    categorizer.fit(docs)
    categorizer.cluster(method=method, n_clusters=n_clusters, random_state=42)

    meta_file = os.path.join(BASE_DIR, "data", "metadata.json")
    multimedia_sys = MultimediaRetrievalSystem(meta_file, base_dir=BASE_DIR)
    return categorizer, multimedia_sys, docs


# Application Header
st.title("🔎 Information Retrieval Systems — Test Set A")
st.caption("Implementation of Automatic Document Categorization & Multimedia Search and Retrieval")

tabs = st.tabs([
    "📑 Q1: Automatic Document Categorization",
    "🖼️ Q2: Multimedia Search & Retrieval",
    "📚 Architectural & Theoretical Notes"
])


# ==============================================================================
# TAB 1: Q1. Automatic Document Categorization System
# ==============================================================================
with tabs[0]:
    st.header("Q1. Automatic Document Categorization System")
    st.markdown(
        """
        This system converts text documents into numerical representations using **TF-IDF**,
        clusters them using **K-Means** or **Hierarchical (Agglomerative)** clustering,
        displays the grouped clusters, and matches user queries to the most relevant cluster and documents.
        """
    )

    # Sidebar controls for Q1
    st.sidebar.markdown("### ⚙️ Q1 Parameters")
    cluster_method = st.sidebar.selectbox(
        "Clustering Technique",
        ["kmeans", "hierarchical"],
        format_func=lambda x: "K-Means Clustering" if x == "kmeans" else "Hierarchical (Agglomerative)",
        key="q1_method"
    )
    n_clusters = st.sidebar.slider("Number of Clusters (K)", min_value=2, max_value=8, value=5, key="q1_k")
    ngram_choice = st.sidebar.radio("N-Gram Range", ["Unigrams (1, 1)", "Unigrams & Bigrams (1, 2)"], index=1)
    ngram_max = 2 if "Bigrams" in ngram_choice else 1

    categorizer, multimedia_sys, docs = get_systems(cluster_method, n_clusters, ngram_max)

    # Top Metrics
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Indexed Documents", len(docs))
    col_m2.metric("TF-IDF Vocabulary Size", len(categorizer.feature_names))
    col_m3.metric("Selected Clusters (K)", n_clusters)
    col_m4.metric("Clustering Technique", cluster_method.upper())

    st.markdown("---")

    # Section 1: Query and Relevance Identification
    st.subheader("🎯 Step 1: Query & Relevant Cluster / Document Identification")
    st.markdown("Enter any search query to project it into the TF-IDF space and identify the best cluster and documents:")

    sample_q_col1, sample_q_col2, sample_q_col3, sample_q_col4 = st.columns(4)
    preset_query = None
    if sample_q_col1.button("🧠 Deep Learning & AI"):
        preset_query = "artificial intelligence neural networks backpropagation gpu"
    if sample_q_col2.button("❤️ Heart Health & Care"):
        preset_query = "cardiovascular disease heart attack cholesterol and blood pressure"
    if sample_q_col3.button("🔭 Telescopes & Cosmos"):
        preset_query = "james webb space telescope astronomy galaxies dark matter"
    if sample_q_col4.button("📈 Stock Markets & Money"):
        preset_query = "stock market portfolio diversification equity investments and bonds"

    user_query = st.text_input(
        "Enter Query String:",
        value=preset_query if preset_query else "deep learning neural network transformers",
        placeholder="Type query words here...",
    )

    if user_query.strip():
        q_res = categorizer.query(user_query, top_k_docs=5)

        if q_res["status"] == "out_of_vocabulary":
            st.warning("⚠️ None of the query terms were present in the document vocabulary.")
        else:
            best_cid = q_res["best_cluster_id"]
            best_cname = q_res["best_cluster_name"]
            best_cscore = q_res["best_cluster_score"]

            st.success(
                f"### 🎯 Most Relevant Cluster: **{best_cname}**  \n"
                f"**Centroid Cosine Similarity Score:** `{best_cscore:.4f}`"
            )

            # Similarity across all clusters
            with st.expander("📊 View Cluster Relevance Distribution", expanded=False):
                chart_data = pd.DataFrame([
                    {"Cluster": info["cluster_name"], "Similarity Score": info["score"]}
                    for cid, info in q_res["cluster_scores"].items()
                ]).set_index("Cluster")
                st.bar_chart(chart_data)

            st.markdown("#### 📄 Top Relevant Documents Ranked by Relevance Score:")
            for doc in q_res["ranked_documents"]:
                cluster_badge = "⭐ In Most Relevant Cluster" if doc["is_in_best_cluster"] else "Other Cluster"
                score_pct = int(doc["relevance_score"] * 100)
                with st.container():
                    st.markdown(
                        f"""
                        <div style="padding: 12px; border-radius: 8px; border: 1px solid #444; margin-bottom: 10px; background-color: rgba(255,255,255,0.03);">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <h4 style="margin: 0;">#{doc['rank']} {doc['title']} <code>[{doc['id']}]</code></h4>
                                <span style="font-size: 1.1em; font-weight: bold; color: #4CAF50;">Relevance Score: {doc['relevance_score']:.4f}</span>
                            </div>
                            <p style="margin: 4px 0 8px 0; color: #888;">Assigned to: <b>{doc['cluster_name']}</b> | <span style="color: #64B5F6;">{cluster_badge}</span></p>
                            <p style="margin: 0; font-size: 0.95em;">{doc['snippet']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    st.markdown("---")

    # Section 2: Display Documents Grouped into Clusters
    st.subheader("📂 Step 2: Documents Grouped into Different Clusters")
    groups = categorizer.get_clustered_groups()

    cluster_cols = st.columns(2)
    for idx, g in enumerate(groups):
        with cluster_cols[idx % 2]:
            with st.expander(f"📁 {g['cluster_name']} ({g['doc_count']} docs)", expanded=True):
                keywords = ", ".join([f"`{term}` ({weight:.2f})" for term, weight in g["top_keywords"][:5]])
                st.markdown(f"**Top TF-IDF Keywords:** {keywords}")
                st.markdown("##### Member Documents:")
                for d in g["documents"]:
                    st.markdown(
                        f"- **{d['title']}** (`{d['id']}`) — *Centroid Similarity: {d['similarity_to_centroid']:.4f}*\n"
                        f"  <span style='color: gray; font-size: 0.85em;'>{d['snippet'][:110]}...</span>",
                        unsafe_allow_html=True
                    )

    st.markdown("---")

    # Section 3: 2D PCA Cluster Scatter Plot
    st.subheader("🗺️ Step 3: 2D TF-IDF Semantic Space Projection (PCA)")
    proj = categorizer.get_2d_projection(query_text=user_query if user_query.strip() else None)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#1e2130")

    coords = proj["doc_coords"]
    labels = proj["cluster_labels"]
    cmap = plt.get_cmap("tab10")

    for k in range(n_clusters):
        mask = [i for i, lbl in enumerate(labels) if lbl == k]
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=120,
            color=cmap(k % 10),
            label=f"Cluster {k+1}",
            alpha=0.85,
            edgecolors="white",
            linewidth=0.8
        )

    # Document annotations
    for i, title in enumerate(proj["titles"]):
        short_title = title.split()[0] + ("..." if len(title.split()) > 1 else "")
        ax.annotate(
            short_title,
            (coords[i, 0], coords[i, 1]),
            fontsize=8,
            color="#e0e0e0",
            xytext=(4, 4),
            textcoords="offset points"
        )

    # Plot Centroids
    centroids = proj["centroid_coords"]
    if len(centroids) > 0:
        ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            s=250,
            color="#FFD700",
            marker="X",
            label="Cluster Centroids",
            edgecolors="black",
            linewidth=1.5
        )

    # Plot Query if available
    if proj["query_coord"] is not None:
        q_c = proj["query_coord"]
        ax.scatter(
            [q_c[0]],
            [q_c[1]],
            s=350,
            color="#FF1493",
            marker="*",
            label=f"Query: '{user_query[:20]}'",
            edgecolors="white",
            linewidth=1.5
        )

    ax.set_title("TF-IDF Vector Space (PCA 2D Projection)", color="white", fontsize=14)
    ax.set_xlabel("Principal Component 1", color="white")
    ax.set_ylabel("Principal Component 2", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#262730", edgecolor="gray", labelcolor="white", loc="best")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ==============================================================================
# TAB 2: Q2. Multimedia Search and Retrieval System
# ==============================================================================
with tabs[1]:
    st.header("Q2. Multimedia Search and Retrieval System")
    st.markdown(
        """
        This system stores and retrieves **text documents** and **images** from a searchable collection.
        It maintains metadata (*title, description, category, upload date, tags*),
        supports **keyword-based searching** of metadata, and implements **image retrieval using color histograms**.
        """
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Q2 Parameters")
    metric_choice = st.sidebar.selectbox(
        "Image Histogram Metric",
        ["intersection", "cosine", "bhattacharyya"],
        format_func=lambda x: {
            "intersection": "Histogram Intersection (Swain & Ballard)",
            "cosine": "Cosine Similarity",
            "bhattacharyya": "Bhattacharyya Coefficient",
        }[x],
        key="q2_metric"
    )
    top_k_results = st.sidebar.slider("Max Results to Return", 1, 15, 6, key="q2_topk")

    # Search Mode Selector
    search_mode = st.radio(
        "Search Modality",
        ["Keyword-based Metadata Search", "Image Retrieval via Color Histogram", "Multimodal (Keyword + Image)"],
        horizontal=True
    )

    query_text_input = ""
    query_image_input = None
    query_image_path = None

    col_q1, col_q2 = st.columns([1, 1])

    if search_mode in ("Keyword-based Metadata Search", "Multimodal (Keyword + Image)"):
        with col_q1:
            st.subheader("🔤 Keyword Search")
            kw_presets = st.columns(3)
            kw_val = ""
            if kw_presets[0].button("Cardiology Heart"):
                kw_val = "cardiovascular heart health cholesterol"
            if kw_presets[1].button("Ocean Water"):
                kw_val = "ocean azure marine water"
            if kw_presets[2].button("Fintech Blockchain"):
                kw_val = "fintech blockchain cryptocurrency"

            query_text_input = st.text_input(
                "Enter metadata keywords:",
                value=kw_val if kw_val else "space astronomy galaxies",
                placeholder="Search title, description, category, tags..."
            )

            # Metadata filters
            filter_col1, filter_col2 = st.columns(2)
            media_filter = filter_col1.selectbox("Filter Media Type", ["all", "text", "image"])
            all_categories = sorted(list(set(it.get("category", "") for it in multimedia_sys.items if it.get("category"))))
            category_filter = filter_col2.selectbox("Filter Category", ["All"] + all_categories)
            cat_val = None if category_filter == "All" else category_filter

    if search_mode in ("Image Retrieval via Color Histogram", "Multimodal (Keyword + Image)"):
        with col_q2:
            st.subheader("🎨 Image Retrieval by Color Histogram")
            image_source = st.radio("Image Query Source", ["Preset Test Query Images", "Upload Custom Image"], horizontal=True)

            if image_source == "Preset Test Query Images":
                preset_img_choice = st.selectbox(
                    "Select a sample query image:",
                    [
                        ("data/queries/query_green_leaf.png", "🍃 Green Botanical Leaf (Tests Green Detection)"),
                        ("data/queries/query_blue_water.png", "🌊 Blue Lagoon Water (Tests Blue Detection)"),
                        ("data/queries/query_red_fire.png", "🔥 Red Campfire Embers (Tests Red Detection)"),
                    ],
                    format_func=lambda x: x[1]
                )
                query_image_path = os.path.join(BASE_DIR, preset_img_choice[0])
                if os.path.exists(query_image_path):
                    query_image_input = Image.open(query_image_path)
                    st.image(query_image_input, caption=preset_img_choice[1], width=200)
            else:
                uploaded_img = st.file_uploader("Upload Query Image (PNG/JPG)", type=["png", "jpg", "jpeg"])
                if uploaded_img is not None:
                    query_image_input = Image.open(uploaded_img)
                    st.image(query_image_input, caption="Uploaded Query Image", width=200)

    st.markdown("---")

    # Execute Search
    st.subheader("📋 Search Results")

    results = []
    if search_mode == "Keyword-based Metadata Search":
        results = multimedia_sys.search_metadata(
            query=query_text_input,
            category_filter=cat_val if "cat_val" in locals() else None,
            media_type_filter=media_filter if "media_filter" in locals() else "all",
            top_k=top_k_results
        )
    elif search_mode == "Image Retrieval via Color Histogram":
        if query_image_input is not None:
            results = multimedia_sys.retrieve_similar_images(
                query_image_input=query_image_input,
                top_k=top_k_results,
                metric=metric_choice
            )
        else:
            st.info("Please select or upload a query image.")
    elif search_mode == "Multimodal (Keyword + Image)":
        results = multimedia_sys.unified_search(
            query_text=query_text_input,
            query_image_input=query_image_input,
            media_type=media_filter if "media_filter" in locals() else "all",
            category_filter=cat_val if "cat_val" in locals() else None,
            top_k=top_k_results
        )

    if results:
        st.write(f"Showing **{len(results)}** matching items ranked by relevance:")
        for rank, item in enumerate(results, start=1):
            with st.container():
                r_col1, r_col2 = st.columns([1, 4])
                media_type = item.get("media_type", "text")

                with r_col1:
                    if media_type == "image":
                        fpath = os.path.join(BASE_DIR, item.get("file_path", ""))
                        if os.path.exists(fpath):
                            st.image(fpath, use_container_width=True)
                        else:
                            st.write("🖼️ [Image File]")
                    else:
                        st.markdown(
                            """
                            <div style="background-color: rgba(255,255,255,0.06); padding: 30px; text-align: center; border-radius: 8px;">
                                <span style="font-size: 2.5em;">📄</span><br>
                                <b>Text Document</b>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                with r_col2:
                    score = item.get("relevance_score", 0.0)
                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: space-between; align-items: baseline;">
                            <h3 style="margin: 0;">#{rank} {item.get('title')}</h3>
                            <span style="font-size: 1.2em; font-weight: bold; color: #4CAF50;">Relevance: {score:.4f}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.progress(min(1.0, float(score)))

                    # Metadata line
                    st.markdown(
                        f"**ID:** `{item.get('id')}` | **Category:** `{item.get('category')}` | "
                        f"**Uploaded:** `{item.get('upload_date')}` | **Path:** `{item.get('file_path')}`"
                    )

                    if item.get("tags"):
                        tags_html = " ".join([f"<span style='background-color: #333; padding: 2px 8px; border-radius: 4px; font-size: 0.85em; margin-right: 4px;'>#{t}</span>" for t in item["tags"]])
                        st.markdown(f"**Tags:** {tags_html}", unsafe_allow_html=True)

                    st.markdown(f"**Description:** {item.get('description')}")

                    if "matched_terms" in item:
                        st.caption(f"Matched keyword terms: {', '.join(item['matched_terms'])}")
                    if "text_score" in item and "visual_score" in item:
                        st.caption(f"Score Fusion: Keyword Match = {item['text_score']:.4f} | Color Similarity = {item['visual_score']:.4f}")

                st.markdown("---")
    else:
        st.info("No matching records found. Try modifying your query.")

    # Color Histogram Visualization
    if query_image_input is not None:
        with st.expander("📊 View Color Histogram Spectrum", expanded=False):
            st.markdown("##### 48-Dimensional Normalized RGB Color Histogram (16 bins per channel):")
            query_hist = multimedia_sys.extract_color_histogram(query_image_input, bins_per_channel=16)

            fig, ax = plt.subplots(figsize=(10, 3.5))
            fig.patch.set_facecolor("#0e1117")
            ax.set_facecolor("#1e2130")

            r_bins = query_hist[:16]
            g_bins = query_hist[16:32]
            b_bins = query_hist[32:]

            x_r = np.arange(16)
            x_g = np.arange(16, 32)
            x_b = np.arange(32, 48)

            ax.bar(x_r, r_bins, color="red", alpha=0.8, label="Red Channel (16 Bins)")
            ax.bar(x_g, g_bins, color="green", alpha=0.8, label="Green Channel (16 Bins)")
            ax.bar(x_b, b_bins, color="blue", alpha=0.8, label="Blue Channel (16 Bins)")

            ax.set_title("Query Image RGB Color Histogram Distribution", color="white")
            ax.set_xlabel("Bin Index (0-47)", color="white")
            ax.set_ylabel("Normalized Frequency", color="white")
            ax.tick_params(colors="white")
            ax.legend(facecolor="#262730", labelcolor="white")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


# ==============================================================================
# TAB 3: Theoretical Foundations & Architecture
# ==============================================================================
with tabs[2]:
    st.header("📚 System Architecture & Theoretical Foundations")

    st.markdown(
        r"""
        ### 1. TF-IDF Numerical Representation (Q1)
        The **Term Frequency - Inverse Document Frequency (TF-IDF)** scheme converts unstructured text into normalized numerical vectors:
        $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
        - **Sublinear Term Frequency:** $\text{TF}(t, d) = 1 + \log(\text{tf}(t, d))$ for $\text{tf} > 0$
        - **Inverse Document Frequency:** $\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + \text{DF}(t)}\right) + 1$
        - **$L_2$ Normalization:** Each document vector $v_d$ is projected onto the unit hypersphere:
          $$\hat{v}_d = \frac{v_d}{\|v_d\|_2}$$

        ---

        ### 2. Clustering Techniques (Q1)
        #### K-Means Clustering
        Partitions $N$ document vectors into $K$ disjoint clusters by minimizing the within-cluster sum of squares (inertia):
        $$J = \sum_{k=1}^K \sum_{d \in C_k} \| \hat{v}_d - \mu_k \|^2$$
        Centroid $\mu_k = \frac{1}{|C_k|} \sum_{d \in C_k} \hat{v}_d$ acts as the thematic prototype of the cluster.

        #### Hierarchical Agglomerative Clustering
        A bottom-up clustering technique that merges the closest pair of clusters at each iteration.
        Using **Ward Linkage**, the objective is to minimize the increase in total within-cluster variance at each merge.

        ---

        ### 3. Query Relevance & Cosine Similarity (Q1 & Q2)
        Given query vector $q$:
        1. **Most Relevant Cluster:**
           $$k^* = \arg\max_{k \in \{1,\dots,K\}} \cos(q, \mu_k) = \frac{q \cdot \mu_k}{\|q\|_2 \|\mu_k\|_2}$$
        2. **Ranked Relevant Documents:**
           $$\text{Score}(d) = \cos(q, \hat{v}_d) = q \cdot \hat{v}_d \in [0.0, 1.0]$$

        ---

        ### 4. Color Histogram Image Retrieval (Q2)
        Images are represented by a discretized color frequency distribution.
        - **Color Histogram:** For channels $R, G, B$, we compute $B=16$ bins each, yielding a 48-dimensional feature vector $H$.
        - **$L_1$ Normalization:** $\sum_{i=1}^{48} H[i] = 1.0$.
        - **Histogram Intersection (Swain & Ballard):**
          $$K(H_q, H_t) = \sum_{i=1}^{48} \min(H_q[i], H_t[i])$$
          Naturally bounded in $[0.0, 1.0]$, invariant to image dimensions, and robust to minor lighting changes.
        """
    )
