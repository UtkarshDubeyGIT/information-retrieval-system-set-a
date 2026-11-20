"""
Utility module for Information Retrieval Systems.
Handles dataset initialization, document loading, synthetic image generation,
and metadata management.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image, ImageDraw


SAMPLE_DOCUMENTS = [
    # Category: Artificial Intelligence & Tech
    (
        "ai_deep_learning.txt",
        "Deep Learning and Neural Network Architectures",
        "Deep learning models utilize multi-layered artificial neural networks to learn representations from raw data. "
        "Convolutional neural networks excel at spatial feature extraction in image classification, while recurrent architectures "
        "and transformers capture sequential dependencies. Backpropagation algorithms compute gradients using stochastic gradient descent, "
        "optimized across high-performance GPUs and tensor processing units to train billions of neural parameters.",
        "Artificial Intelligence",
        ["deep learning", "neural networks", "backpropagation", "transformers", "gpu", "ai"]
    ),
    (
        "nlp_transformers.txt",
        "Natural Language Processing and Large Language Models",
        "Natural language processing has undergone a revolution with self-attention transformer models. "
        "Tokenization strategies, attention heads, and positional encodings enable models to process full contexts simultaneously. "
        "Techniques such as masked language modeling, causal autoregression, and instruction fine-tuning have made conversational "
        "chatbots and semantic search systems highly capable across multilingual tasks.",
        "Artificial Intelligence",
        ["nlp", "transformers", "language models", "chatbots", "tokenization", "text analysis"]
    ),
    (
        "cloud_microservices.txt",
        "Cloud Infrastructure and Containerized Microservices",
        "Modern cloud native applications break monolithic software into modular microservices. "
        "Container virtualization tools like Docker and container orchestration platforms like Kubernetes automate deployment, "
        "horizontal auto-scaling, and self-healing. Cloud providers offer serverless functions, managed relational databases, "
        "and distributed caching to support high availability and fault-tolerant distributed web systems.",
        "Information Technology",
        ["cloud computing", "docker", "kubernetes", "microservices", "devops", "containers"]
    ),

    # Category: Medicine & Health
    (
        "cardiovascular_health.txt",
        "Cardiovascular Health and Heart Disease Prevention",
        "Cardiovascular diseases remain the leading cause of global mortality. Coronary artery disease involves atherosclerotic "
        "plaque buildup inside arterial walls, restricting blood flow to the myocardial muscle. Clinical management includes "
        "monitoring systolic blood pressure, lowering LDL cholesterol with statins, regular aerobic exercise, and emergency "
        "interventions such as angioplasty and coronary artery bypass surgery.",
        "Medicine & Health",
        ["heart disease", "cardiovascular", "cholesterol", "blood pressure", "cardiology", "health"]
    ),
    (
        "immunology_vaccines.txt",
        "Immunology, Antibodies, and Modern Vaccine Platforms",
        "The human immune system relies on innate and adaptive responses against invading pathogens. "
        "B lymphocytes produce antigen-specific neutralizing antibodies, while cytotoxic T lymphocytes eliminate infected host cells. "
        "Modern vaccine technologies, including messenger RNA and recombinant viral vector platforms, safely present antigen codes to stimulate "
        "long-lasting immunological memory without causing active disease.",
        "Medicine & Health",
        ["vaccines", "immunology", "antibodies", "pathogens", "immune response", "viruses"]
    ),
    (
        "neuroscience_brain.txt",
        "Neuroscience and Cerebral Cognitive Function",
        "The human brain comprises billions of neurons communicating across synaptic junctions via chemical neurotransmitters. "
        "Neuroplasticity enables the cerebral cortex to reorganize neural pathways during learning, memory consolidation, and recovery "
        "from trauma. Modern functional magnetic resonance imaging provides non-invasive windows into cognitive perception and neurological disorders.",
        "Medicine & Health",
        ["neuroscience", "brain", "neurons", "cognition", "synaptic", "neurology"]
    ),

    # Category: Space Science & Astronomy
    (
        "astronomy_telescopes.txt",
        "Deep Space Astronomy and Space Telescopes",
        "Spaceborne observatories like the James Webb Space Telescope observe the cosmos in the infrared spectrum. "
        "Astronomers peer back over thirteen billion years to detect the earliest primordial galaxies forming after the Big Bang. "
        "Infrared spectroscopic analysis reveals atmospheric chemical signatures on distant exoplanets and maps cosmological dark matter halos.",
        "Space Science",
        ["astronomy", "telescopes", "space", "galaxies", "james webb", "cosmology"]
    ),
    (
        "mars_planetary_science.txt",
        "Mars Planetary Science and Robotic Exploration",
        "Exploration of the red planet Mars has accelerated with autonomous rovers equipped with drill spectrometers and hazard cameras. "
        "Planetary geologists study ancient lakebeds in Jezero Crater searching for biosignature fossils. "
        "In situ resource utilization, extracting oxygen from the atmospheric carbon dioxide, paves the way for human planetary exploration.",
        "Space Science",
        ["mars", "planetary exploration", "rovers", "spaceflight", "geology", "space"]
    ),
    (
        "black_holes_gravity.txt",
        "Black Holes, General Relativity, and Gravitational Waves",
        "General relativity predicts extreme gravitational curvature where matter collapses into a spacetime singularity. "
        "Black holes feature an event horizon beyond which neither radiation nor matter can escape. "
        "Laser interferometer observatories detect gravitational ripples produced when binary black hole systems merge across cosmic distances.",
        "Space Science",
        ["black holes", "gravity", "general relativity", "gravitational waves", "astrophysics"]
    ),

    # Category: Finance & Economics
    (
        "macroeconomics_inflation.txt",
        "Macroeconomic Policy, Inflation, and Central Banking",
        "Macroeconomic stability requires balancing economic growth with sustainable inflation targets. "
        "Central banking institutions use quantitative tightening and benchmark interest rate adjustments to curb rising consumer price indices. "
        "Global trade dynamics, labor supply curves, fiscal stimulus deficits, and commodity supply chain shocks influence national gross domestic product.",
        "Finance & Economics",
        ["macroeconomics", "inflation", "central bank", "interest rates", "monetary policy", "economy"]
    ),
    (
        "stock_market_investing.txt",
        "Equity Markets, Portfolio Diversification, and Asset Allocation",
        "Investing in public capital markets requires disciplined risk management and portfolio diversification. "
        "Modern portfolio theory suggests blending non-correlated asset classes, such as equities, sovereign bonds, and real estate, "
        "to optimize Sharpe ratios. Fundamental equity valuation inspects price-to-earnings ratios, free cash flow margins, and dividend sustainability.",
        "Finance & Economics",
        ["stocks", "investing", "portfolio", "finance", "equities", "asset management"]
    ),
    (
        "crypto_fintech.txt",
        "Financial Technology and Decentralized Blockchain Ledgers",
        "Fintech innovations transform payment rails and banking infrastructure through distributed cryptographic ledgers. "
        "Blockchain protocols utilize consensus algorithms such as proof-of-stake to record immutable digital transactions without central intermediaries. "
        "Decentralized finance protocols enable programmable algorithmic lending, tokenized assets, and automated market maker liquidity pools.",
        "Finance & Economics",
        ["fintech", "blockchain", "cryptocurrency", "defi", "finance", "transactions"]
    ),

    # Category: Sports & Athletics
    (
        "football_tactics.txt",
        "Association Football Strategy and Tactical Pressing",
        "Professional soccer tactical systems emphasize high pressing, spatial awareness, and fluid positional transitions. "
        "Coaches deploy 4-3-3 or 3-5-2 formations to dominate midfield possession, create numerical overloads in half-spaces, "
        "and maintain defensive compactness. Modern match analytics track expected goals, defensive recovery distance, and set-piece efficiency.",
        "Sports & Athletics",
        ["football", "soccer", "tactics", "sports", "world cup", "athletics"]
    ),
    (
        "tennis_grand_slam.txt",
        "Grand Slam Tennis Championships and Court Surfaces",
        "Elite tennis competition tests athletic endurance, technical ball striking, and psychological resilience across four Grand Slam tournaments. "
        "Playing surfaces alter ball bounce dynamics drastically, from high-friction slow European clay to lightning-fast Wimbledon grass courts. "
        "Players balance heavy topspin groundstrokes with powerful serves exceeding two hundred kilometers per hour.",
        "Sports & Athletics",
        ["tennis", "grand slam", "wimbledon", "sports", "championship", "athletics"]
    ),
    (
        "marathon_endurance.txt",
        "Marathon Training, Aerobic Physiology, and Endurance Running",
        "Distance running performance relies heavily on aerobic capacity, lactate threshold efficiency, and running economy. "
        "Marathon runners structure training blocks around progressive long runs, interval speed sessions, and glycogen replenishment nutrition. "
        "Carbohydrate loading protocols and electrolyte hydration balance maintain muscle glycogen stores over the demanding 42.195 kilometer distance.",
        "Sports & Athletics",
        ["marathon", "endurance", "running", "athletics", "training", "sports"]
    ),
]


CATALOG_IMAGES = [
    {
        "filename": "forest_lush_green.png",
        "title": "Lush Emerald Forest Canopy",
        "description": "Dense woodland canopy filled with vivid green pine leaves, moss-covered trees, and vibrant emerald flora.",
        "category": "Nature & Landscape",
        "tags": ["nature", "forest", "green", "trees", "foliage", "landscape"],
        "primary_color": (34, 139, 34),    # Forest green
        "secondary_color": (50, 205, 50),   # Lime green
        "accent_color": (107, 142, 35),     # Olive drab
        "pattern": "gradient_vertical"
    },
    {
        "filename": "ocean_deep_blue.png",
        "title": "Deep Azure Ocean Waters",
        "description": "Vibrant cobalt and azure ocean waves under clear maritime skies, exhibiting rich blue aquatic hues.",
        "category": "Nature & Landscape",
        "tags": ["ocean", "sea", "blue", "water", "marine", "azure"],
        "primary_color": (0, 105, 148),     # Sea blue
        "secondary_color": (30, 144, 255),  # Dodger blue
        "accent_color": (0, 206, 209),      # Dark turquoise
        "pattern": "gradient_radial"
    },
    {
        "filename": "sunset_crimson_red.png",
        "title": "Fiery Crimson Horizon Sunset",
        "description": "Dramatic sunset horizon bursting with blazing crimson reds, rich scarlet, and warm glowing amber light.",
        "category": "Nature & Landscape",
        "tags": ["sunset", "red", "crimson", "sky", "horizon", "warmth"],
        "primary_color": (220, 20, 60),     # Crimson
        "secondary_color": (255, 69, 0),    # Red orange
        "accent_color": (178, 34, 34),      # Firebrick
        "pattern": "gradient_vertical"
    },
    {
        "filename": "desert_golden_yellow.png",
        "title": "Golden Sahara Sand Dunes",
        "description": "Expansive desert dunes glowing with warm golden yellow sand, amber ridges, and sunny ochre minerals.",
        "category": "Nature & Landscape",
        "tags": ["desert", "sand", "yellow", "golden", "dunes", "sahara"],
        "primary_color": (238, 201, 0),     # Gold
        "secondary_color": (244, 164, 96),  # Sandy brown
        "accent_color": (218, 165, 32),     # Goldenrod
        "pattern": "waves"
    },
    {
        "filename": "galaxy_neon_purple.png",
        "title": "Cosmic Purple Nebula Cloud",
        "description": "Astronomical deep space nebula glowing with cosmic purple, ultraviolet dust clouds, and magenta stars.",
        "category": "Space Science",
        "tags": ["galaxy", "nebula", "space", "purple", "violet", "astronomy"],
        "primary_color": (128, 0, 128),     # Purple
        "secondary_color": (148, 0, 211),   # Dark violet
        "accent_color": (75, 0, 130),       # Indigo
        "pattern": "gradient_radial"
    }
]

QUERY_IMAGES = [
    {
        "filename": "query_green_leaf.png",
        "title": "Green Botanical Specimen",
        "description": "Bright green botanical leaf specimen displaying lush emerald and lime chlorophyll tones.",
        "category": "Nature & Landscape",
        "tags": ["green", "leaf", "plant", "botanical"],
        "primary_color": (46, 139, 87),     # Sea green
        "secondary_color": (60, 179, 113),  # Medium sea green
        "accent_color": (144, 238, 144),    # Light green
        "pattern": "gradient_vertical"
    },
    {
        "filename": "query_blue_water.png",
        "title": "Pure Blue Lagoon Water",
        "description": "Calm coastal lagoon water surface showing deep blue and cyan aquatic reflections.",
        "category": "Nature & Landscape",
        "tags": ["blue", "water", "lagoon", "sea"],
        "primary_color": (25, 25, 112),     # Midnight blue
        "secondary_color": (0, 0, 205),      # Medium blue
        "accent_color": (65, 105, 225),     # Royal blue
        "pattern": "gradient_radial"
    },
    {
        "filename": "query_red_fire.png",
        "title": "Glowing Red Campfire Coals",
        "description": "Intense glowing campfire embers radiating rich crimson, scarlet, and ruby heat.",
        "category": "Nature & Landscape",
        "tags": ["red", "fire", "crimson", "embers"],
        "primary_color": (205, 0, 0),       # Deep red
        "secondary_color": (255, 36, 0),    # Scarlet
        "accent_color": (139, 0, 0),        # Dark red
        "pattern": "gradient_vertical"
    }
]


def create_synthetic_image(info: dict, filepath: str, size: Tuple[int, int] = (256, 256)):
    """Creates a high quality synthetic image with specified color profiles and patterns."""
    width, height = size
    img = Image.new("RGB", (width, height), info["primary_color"])
    draw = ImageDraw.Draw(img)

    p_col = np.array(info["primary_color"], dtype=float)
    s_col = np.array(info["secondary_color"], dtype=float)
    a_col = np.array(info["accent_color"], dtype=float)

    pattern = info.get("pattern", "gradient_vertical")

    if pattern == "gradient_vertical":
        for y in range(height):
            ratio = y / float(height - 1)
            color = (1.0 - ratio) * p_col + ratio * s_col
            # Add slight noise/texture
            color = np.clip(color, 0, 255).astype(int)
            draw.line([(0, y), (width, y)], fill=tuple(color))
        # Draw central accent oval
        draw.ellipse([width * 0.25, height * 0.25, width * 0.75, height * 0.75], outline=tuple(a_col.astype(int)), width=4)

    elif pattern == "gradient_radial":
        cx, cy = width / 2.0, height / 2.0
        max_dist = np.sqrt(cx**2 + cy**2)
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                t = min(1.0, dist / max_dist)
                c = (1.0 - t) * s_col + t * p_col
                arr[y, x] = np.clip(c, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, "RGB")
        draw = ImageDraw.Draw(img)
        draw.ellipse([width * 0.3, height * 0.3, width * 0.7, height * 0.7], fill=tuple(a_col.astype(int)))

    elif pattern == "waves":
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                wave = np.sin(x / 15.0) * 15.0 + np.cos(y / 20.0) * 10.0
                t = ((y + wave) % height) / float(height)
                c = (1.0 - t) * p_col + t * s_col * 0.8 + a_col * 0.2
                arr[y, x] = np.clip(c, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, "RGB")

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    img.save(filepath, format="PNG")


def initialize_sample_dataset(base_dir: str = ".") -> Tuple[str, str, str]:
    """
    Initializes documents, images, query images, and metadata inside the data directory.
    Returns paths to (documents_dir, images_dir, metadata_path).
    """
    data_dir = os.path.join(base_dir, "data")
    docs_dir = os.path.join(data_dir, "documents")
    images_dir = os.path.join(data_dir, "images")
    queries_dir = os.path.join(data_dir, "queries")
    metadata_path = os.path.join(data_dir, "metadata.json")

    os.makedirs(docs_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(queries_dir, exist_ok=True)

    metadata_records = []

    # 1. Write text documents
    for idx, (fname, title, content, cat, tags) in enumerate(SAMPLE_DOCUMENTS, start=1):
        fpath = os.path.join(docs_dir, fname)
        rel_path = os.path.relpath(fpath, base_dir)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content)

        upload_date = f"2026-{(idx % 12) + 1:02d}-{(idx * 2) % 28 + 1:02d}"
        metadata_records.append({
            "id": f"DOC-{idx:03d}",
            "title": title,
            "description": content[:180] + "...",
            "category": cat,
            "upload_date": upload_date,
            "media_type": "text",
            "file_path": rel_path,
            "tags": tags
        })

    # 2. Write catalog images
    for idx, img_info in enumerate(CATALOG_IMAGES, start=1):
        fpath = os.path.join(images_dir, img_info["filename"])
        rel_path = os.path.relpath(fpath, base_dir)
        create_synthetic_image(img_info, fpath)

        upload_date = f"2026-{(idx % 12) + 1:02d}-{(idx * 3) % 28 + 1:02d}"
        metadata_records.append({
            "id": f"IMG-{idx:03d}",
            "title": img_info["title"],
            "description": img_info["description"],
            "category": img_info["category"],
            "upload_date": upload_date,
            "media_type": "image",
            "file_path": rel_path,
            "tags": img_info["tags"]
        })

    # 3. Write query images (stored for testing/querying)
    for q_info in QUERY_IMAGES:
        fpath = os.path.join(queries_dir, q_info["filename"])
        create_synthetic_image(q_info, fpath)

    # 4. Save metadata.json
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_records, f, indent=2)

    return docs_dir, images_dir, metadata_path


def load_documents_from_directory(directory: str) -> List[Dict[str, str]]:
    """Loads all .txt files from a directory into a list of dicts with id, filename, content."""
    docs = []
    if not os.path.exists(directory):
        return docs
    for fname in sorted(os.listdir(directory)):
        if fname.endswith(".txt"):
            fpath = os.path.join(directory, fname)
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()
            docs.append({
                "id": fname,
                "filename": fname,
                "title": fname.replace(".txt", "").replace("_", " ").title(),
                "content": content
            })
    return docs


def load_metadata_store(metadata_path: str) -> List[Dict]:
    """Loads metadata records from JSON file."""
    if not os.path.exists(metadata_path):
        return []
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    d, i, m = initialize_sample_dataset(".")
    print(f"Sample dataset initialized:\n- Documents: {d}\n- Images: {i}\n- Metadata: {m}")
