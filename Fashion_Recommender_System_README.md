# 👗 Fashion Recommender System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fashion-recommender-system-04.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-ResNet50-D00000?logo=keras&logoColor=white)](https://keras.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-NearestNeighbors-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-yellow)](https://huggingface.co/joel799659)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, deep learning-powered **Visual Fashion Recommendation System** that leverages **Transfer Learning (ResNet-50)** and **k-Nearest Neighbors (k-NN)** to discover and recommend visually similar apparel, footwear, and accessories from an uploaded image in real time.

🔗 **Live Demo:** [fashion-recommender-system-04.streamlit.app](https://fashion-recommender-system-04.streamlit.app/)  
📦 **GitHub Repository:** [joel799659/Fashion-recommender-system](https://github.com/joel799659/Fashion-recommender-system)  
🤗 **Hugging Face Assets:** [Embeddings Dataset](https://huggingface.co/datasets/joel799659/fashion-recommender-embeddings-pkl) | [Images Dataset](https://huggingface.co/datasets/joel799659/fashion-recommender-images)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture & Workflow](#-system-architecture--workflow)
- [How It Works (Technical Details)](#-how-it-works-technical-details)
- [Tech Stack](#-tech-stack)
- [Repository Structure](#-repository-structure)
- [Getting Started & Local Installation](#-getting-started--local-installation)
- [Dataset & Cloud Storage Architecture](#-dataset--cloud-storage-architecture)
- [Roadmap & Future Enhancements](#-roadmap--future-enhancements)
- [Author & Acknowledgements](#-author--acknowledgements)
- [License](#-license)

---

## 🌟 Overview

In modern e-commerce platforms, text-based search queries often fail when users cannot describe intricate patterns, cuts, textures, or silhouettes. The **Fashion Recommender System** bridges this semantic gap by allowing users to upload a query photograph of an item they love.

The system extracts deep semantic feature vectors using a convolutional neural network pre-trained on ImageNet (**ResNet-50** with **Global Max Pooling**), normalizes the representations, and performs fast similarity retrieval via **Euclidean Distance** to display the top 5 closest matches.

---

## ✨ Key Features

- **🧠 Deep Visual Embeddings:** Employs a pre-trained **ResNet-50** deep convolutional backbone to capture fine-grained fashion features (color, texture, pattern, shape) as high-dimensional (2048-D) vectors.
- **📐 L2 Normalized Representations:** Uses Euclidean L2 normalization to align representations for fast and invariant metric calculation.
- **⚡ Fast Metric Retrieval:** Uses Scikit-Learn's `NearestNeighbors` brute-force Euclidean distance search across tens of thousands of items to find nearest visual neighbors in milliseconds.
- **☁️ Cloud-Native Asset Pipeline (Hugging Face Datasets):** Solves the challenge of GitHub's 100MB file limit by hosting precomputed 365MB embeddings and catalog images on Hugging Face Hub, dynamically downloaded and cached in Streamlit Cloud.
- **🖥️ Interactive Streamlit UI:** Clean, responsive drag-and-drop web interface with instant visual feedback and organized column layouts for recommended items.
- **🧪 Modular Pipeline:** Separated batch offline feature indexing (`app.py`), local OpenCV experimentation (`test.py`), and cloud-ready web serving (`main.py`).

---

## 🏗️ System Architecture & Workflow

```
   ┌─────────────────────────────────────────────────────────────┐
   │                OFFLINE INDEXING PIPELINE                    │
   └─────────────────────────────────────────────────────────────┘
          Images Catalog (44,000+ Fashion Products)
                            │
                            ▼
           Preprocessing (224x224, ResNet Preprocess)
                            │
                            ▼
        ResNet50 Backbone (Weights: ImageNet, Top: False)
                            │
                            ▼
                 Global MaxPooling 2D (2048-D)
                            │
                            ▼
                   L2 Vector Normalization
                            │
                            ▼
         embedding.pkl (365MB) & filenames_hf.pkl
                            │
                            ▼
               Upload to Hugging Face Hub
```

```
   ┌─────────────────────────────────────────────────────────────┐
   │               ONLINE REAL-TIME INFERENCE                    │
   └─────────────────────────────────────────────────────────────┘
                     User Uploads Query Image
                                │
                                ▼
                       Streamlit Interface
                                │
                                ▼
                   Image Preprocessing (224x224)
                                │
                                ▼
                   ResNet50 Feature Extractor
                                │
                                ▼
                 Normalized 2048-D Query Vector
                                │
                                ▼
              NearestNeighbors (k=6, Euclidean Metric)
                                │
                                ▼
                 Top 5 Nearest Neighbor Indices
                                │
                                ▼
           Fetch URLs from Hugging Face Image Dataset
                                │
                                ▼
              Render Recommendations in Streamlit UI
```

---

## 🔬 How It Works (Technical Details)

### 1. Feature Representation with ResNet-50
The model uses **ResNet-50** (Residual Networks) pre-trained on ImageNet:
- The classification head (`include_top=False`) is discarded to retain generic semantic image features.
- We append a **GlobalMaxPooling2D** layer to condense the 2D feature maps from the final convolutional block into a single 1D vector of length **2,048**.
- The weights are frozen (`trainable = False`), ensuring stable, deterministic feature extraction.

$$\mathbf{x}_{\text{raw}} = \text{GlobalMaxPooling2D}(\text{ResNet50}(\mathbf{I})) \in \mathbb{R}^{2048}$$

### 2. Feature Normalization
To ensure scale-invariant metric comparisons, the extracted vector is normalized by its Euclidean L2 norm:

$$\mathbf{x}_{\text{norm}} = \frac{\mathbf{x}_{\text{raw}}}{\|\mathbf{x}_{\text{raw}}\|_2} = \frac{\mathbf{x}_{\text{raw}}}{\sqrt{\sum_{i=1}^{2048} x_i^2}}$$

### 3. Similarity Search via Nearest Neighbors
When comparing normalized vectors, Euclidean distance corresponds directly to Cosine distance:

$$d(\mathbf{u}, \mathbf{v}) = \sqrt{\sum_{i=1}^{2048} (u_i - v_i)^2}$$

The Scikit-Learn `NearestNeighbors(n_neighbors=6, metric='euclidean')` model queries the index matrix $\mathbf{X} \in \mathbb{R}^{N \times 2048}$ to retrieve the closest matching product images.

---

## 🛠️ Tech Stack

| Category | Technologies / Libraries |
|---|---|
| **Deep Learning & Modeling** | [TensorFlow](https://www.tensorflow.org/), [Keras](https://keras.io/) (ResNet50) |
| **Vector Similarity** | [Scikit-Learn](https://scikit-learn.org/) (`NearestNeighbors`), [NumPy](https://numpy.org/) |
| **Frontend & Web Framework** | [Streamlit](https://streamlit.io/) |
| **Dataset & Remote Storage** | [Hugging Face Hub](https://huggingface.co/) (`huggingface_hub`) |
| **Image Processing** | [Pillow (PIL)](https://python-pillow.org/), [OpenCV](https://opencv.org/) |
| **Cloud Hosting** | [Streamlit Community Cloud](https://streamlit.io/cloud) |

---

## 📂 Repository Structure

```
Fashion-recommender-system/
├── app.py              # Batch extraction script to generate embeddings locally
├── main.py             # Streamlit web application (production / cloud version)
├── test.py             # Standalone local script to test recommendations via OpenCV
├── filenames.pkl       # Pickled list of local product image file paths
├── filenames_hf.pkl    # Pickled list of Hugging Face dataset relative image paths
├── embedding.pkl       # Pickled feature vectors matrix (N x 2048) [Hosted on HF]
├── requirements.txt    # Application dependencies for Streamlit and local run
├── sample/             # Sample query images for testing (e.g., jersey.jpg)
│   └── jersey.jpg
├── uploads/            # Temporary directory for user-uploaded query images
└── README.md           # Project documentation and setup guide
```

### File Breakdown:
- **`main.py`**: The production Streamlit application. Automatically fetches `embedding.pkl` and `filenames_hf.pkl` from Hugging Face Hub using `@st.cache_resource`, processes user-uploaded query images, runs k-NN search, and displays the top 5 visually similar apparel items.
- **`app.py`**: Offline script that scans an `images/` directory, extracts normalized ResNet50 embeddings for each photo, and serializes them into `embedding.pkl` and `filenames.pkl`.
- **`test.py`**: A quick development test utility using OpenCV to read `sample/jersey.jpg`, retrieve top matches from `embedding.pkl`, and display them in desktop windows.
- **`requirements.txt`**: Specifies all required libraries for the environment.

---

## 🚀 Getting Started & Local Installation

### Prerequisites
- Python 3.10 or 3.11 installed
- Git installed
- ~4GB RAM recommended for running ResNet-50 inference

### 1. Clone the Repository
```bash
git clone https://github.com/joel799659/Fashion-recommender-system.git
cd Fashion-recommender-system
```

### 2. Create and Activate a Virtual Environment
```bash
# On macOS / Linux:
python3 -m venv venv
source venv/bin/activate

# On Windows:
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run main.py
```
Open your browser and navigate to `http://localhost:8501`.

> **Note:** On first launch, `main.py` will automatically download `embedding.pkl` and `filenames_hf.pkl` from Hugging Face. Subsequent launches will use the cached local copy.

---

## ☁️ Dataset & Cloud Storage Architecture

The system is trained and indexed on the **Fashion Product Images Dataset** (covering apparel, footwear, watches, and accessories).

### The Challenge:
- The full embeddings matrix (`embedding.pkl`) is **~365 MB**, exceeding GitHub's 100 MB single-file limit.
- Hosting tens of thousands of high-resolution images inside a Git repository causes bloat and slow deployment.

### The Solution:
1. **Embeddings:** Uploaded as a dataset repository to Hugging Face:
   - [`joel799659/fashion-recommender-embeddings-pkl`](https://huggingface.co/datasets/joel799659/fashion-recommender-embeddings-pkl)
   - Downloaded on demand at runtime via `hf_hub_download` and cached in memory using `@st.cache_resource`.
2. **Catalog Images:** Uploaded to Hugging Face dataset:
   - [`joel799659/fashion-recommender-images`](https://huggingface.co/datasets/joel799659/fashion-recommender-images)
   - Filenames are stored in `filenames_hf.pkl`, and images are rendered via direct CDN URLs (`https://huggingface.co/datasets/joel799659/fashion-recommender-images/resolve/main/<image_id>.jpg`).

This decouples heavy binary assets from the application logic, allowing continuous deployment on Streamlit Cloud without timeouts or storage breaches.

---

## 💡 Roadmap & Future Enhancements

- [ ] **Multi-Modal Search:** Combine text queries with image queries using OpenAI's **CLIP** (e.g., "Find this shirt but in blue").
- [ ] **Vector Database Migration:** Transition from Scikit-Learn `NearestNeighbors` to **FAISS**, **Qdrant**, or **Milvus** for sub-millisecond retrieval across millions of vectors.
- [ ] **Category & Attribute Filtering:** Add filters for gender, price range, brand, and apparel category (tops, bottoms, footwear).
- [ ] **Bounding Box Object Detection:** Integrate YOLOv8 to automatically detect and crop multiple garments from a full-body outfit image.

---

## 👤 Author & Acknowledgements

**Joel**
- GitHub: [@joel799659](https://github.com/joel799659)
- Live App: [Fashion Recommender System](https://fashion-recommender-system-04.streamlit.app/)

### Acknowledgements:
- [Kaggle Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)
- [Streamlit](https://streamlit.io/) for the web framework.
- [Hugging Face](https://huggingface.co/) for open model and dataset infrastructure.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) - feel free to use and modify it for your own personal and commercial projects.
