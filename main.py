import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from huggingface_hub import hf_hub_download


# =========================================================
# Hugging Face Configuration
# =========================================================

HF_IMAGES_BASE = (
    "https://huggingface.co/datasets/"
    "joel799659/fashion-recommender-images/"
    "resolve/main/"
)


# =========================================================
# Load Embeddings
# =========================================================

@st.cache_resource
def load_embeddings():

    embedding_path = hf_hub_download(
        repo_id="joel799659/fashion-recommender-embeddings-pkl",
        filename="embedding.pkl",
        repo_type="dataset"
    )

    with open(embedding_path, "rb") as file:
        embeddings = pickle.load(file)

    return np.array(embeddings)


feature_list = load_embeddings()


# =========================================================
# Load Image Filenames
# =========================================================

@st.cache_resource
def load_filenames():

    filenames_path = hf_hub_download(
        repo_id="joel799659/fashion-recommender-images",
        filename="filenames_hf.pkl",
        repo_type="dataset"
    )

    with open(filenames_path, "rb") as file:
        filenames = pickle.load(file)

    return filenames


filenames = load_filenames()


# =========================================================
# Load ResNet50 Model
# =========================================================

@st.cache_resource
def load_model():

    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False

    model = tensorflow.keras.Sequential([
        base_model,
        GlobalMaxPooling2D()
    ])

    return model


model = load_model()


# =========================================================
# Streamlit Page
# =========================================================

st.title("Fashion Recommender System")


# =========================================================
# Save Uploaded Image
# =========================================================

def save_uploaded_file(uploaded_file):

    try:

        os.makedirs("uploads", exist_ok=True)

        file_path = os.path.join(
            "uploads",
            uploaded_file.name
        )

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        return file_path

    except Exception as e:

        st.error(
            f"File upload error: {e}"
        )

        return None


# =========================================================
# Feature Extraction
# =========================================================

def feature_extraction(img_path, model):

    img = image.load_img(
        img_path,
        target_size=(224, 224)
    )

    img_array = image.img_to_array(img)

    expanded_img_array = np.expand_dims(
        img_array,
        axis=0
    )

    preprocessed_img = preprocess_input(
        expanded_img_array
    )

    result = model.predict(
        preprocessed_img,
        verbose=0
    ).flatten()

    normalized_result = result / norm(result)

    return normalized_result


# =========================================================
# Recommendation
# =========================================================

def recommend(features, features_list):

    neighbors = NearestNeighbors(
        n_neighbors=6,
        algorithm="brute",
        metric="euclidean"
    )

    neighbors.fit(features_list)

    distances, indices = neighbors.kneighbors(
        [features]
    )

    return indices


# =========================================================
# Hugging Face Image URL
# =========================================================

def get_hf_image_url(path):

    return HF_IMAGES_BASE + path


# =========================================================
# Upload Image
# =========================================================

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


# =========================================================
# Process Uploaded Image
# =========================================================

if uploaded_file is not None:

    uploaded_path = save_uploaded_file(
        uploaded_file
    )

    if uploaded_path:

        # -----------------------------------------
        # Display Uploaded Image
        # -----------------------------------------

        display_image = Image.open(
            uploaded_file
        )

        st.image(
            display_image,
            caption="Uploaded Image",
            width=300
        )


        # -----------------------------------------
        # Extract Features
        # -----------------------------------------

        features = feature_extraction(
            uploaded_path,
            model
        )


        # -----------------------------------------
        # Get Recommendations
        # -----------------------------------------

        indices = recommend(
            features,
            feature_list
        )


        # -----------------------------------------
        # Display Recommendations
        # -----------------------------------------

        st.subheader(
            "Recommended Images"
        )


        col1, col2, col3, col4, col5 = st.columns(5)


        # -----------------------------------------
        # Recommendation 1
        # -----------------------------------------

        with col1:

            st.image(
                get_hf_image_url(
                    filenames[indices[0][0]]
                ),
                width=150
            )


        # -----------------------------------------
        # Recommendation 2
        # -----------------------------------------

        with col2:

            st.image(
                get_hf_image_url(
                    filenames[indices[0][1]]
                ),
                width=150
            )


        # -----------------------------------------
        # Recommendation 3
        # -----------------------------------------

        with col3:

            st.image(
                get_hf_image_url(
                    filenames[indices[0][2]]
                ),
                width=150
            )


        # -----------------------------------------
        # Recommendation 4
        # -----------------------------------------

        with col4:

            st.image(
                get_hf_image_url(
                    filenames[indices[0][3]]
                ),
                width=150
            )


        # -----------------------------------------
        # Recommendation 5
        # -----------------------------------------

        with col5:

            st.image(
                get_hf_image_url(
                    filenames[indices[0][4]]
                ),
                width=150
            )


    else:

        st.error(
            "Some error occurred while uploading the file."
        )
