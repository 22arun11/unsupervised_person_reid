import streamlit as st
# import cv2 # No longer strictly needed if only using PIL
import numpy as np
import torch
from PIL import Image, ImageOps # Added ImageOps for potential padding
import io
from sklearn.metrics.pairwise import cosine_similarity # euclidean_distances removed for simplicity now
from sklearn.cluster import KMeans
import os
import pickle
from transformers import ViTFeatureExtractor, ViTModel
import time
import shutil

# --- Page Config (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(layout="wide")

# --- Configuration ---
NUM_INITIAL_PHOTOS = 3
TARGET_IMG_SIZE = (224, 224) # Size expected by ViT feature extractor
DATA_DIR = "./data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
FEATURES_FILE = os.path.join(DATA_DIR, "features.npy")
IMAGE_PATHS_FILE = os.path.join(DATA_DIR, "image_paths.pkl")
CLUSTER_MODEL_FILE = os.path.join(DATA_DIR, "kmeans_model.pkl")
PERSON_ID_COUNTER_FILE = os.path.join(DATA_DIR, "person_id_counter.txt")
# Adjust threshold - combined similarity might behave differently
MATCH_THRESHOLD = 0.65 # Example: Lowered threshold slightly for avg similarity
VIT_EMBEDDING_DIM = 768 # For google/vit-base-patch16-224-in21k

# Create data directories if they don't exist
os.makedirs(IMAGE_DIR, exist_ok=True)

# --- Model Loading (Cached) ---
@st.cache_resource
def load_model():
    """Loads the pre-trained Vision Transformer model and feature extractor."""
    try:
        model_name = "google/vit-base-patch16-224-in21k"
        feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        model = ViTModel.from_pretrained(model_name)
        # Check embedding dimension matches config
        global VIT_EMBEDDING_DIM
        if model.config.hidden_size != VIT_EMBEDDING_DIM:
             st.warning(f"Model hidden size ({model.config.hidden_size}) doesn't match configured VIT_EMBEDDING_DIM ({VIT_EMBEDDING_DIM}). Using model's value.")
             VIT_EMBEDDING_DIM = model.config.hidden_size

        model.eval() # Set to evaluation mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"Transformer Model loaded on device: {device}")
        return feature_extractor, model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# --- Feature Extraction (Split Version) ---
def extract_single_feature_from_image(img_pil, feature_extractor, model, device):
    """Helper function to extract features from a single PIL image."""
    # Preprocess image using the feature extractor
    inputs = feature_extractor(images=img_pil, return_tensors="pt").to(device)
    # Extract features
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the embedding of the [CLS] token
    features = outputs.last_hidden_state[:, 0, :].cpu().numpy() # Shape: (1, embedding_dim)
    return features

def extract_features_transformer_split(image_bytes, feature_extractor, model, device):
    """
    Splits image into upper/lower halves, extracts features for each,
    and returns both feature vectors.
    """
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        width, height = pil_image.size

        # Handle cases where image might be too small to split meaningfully
        if height < 20: # Arbitrary minimum height for splitting
             st.warning("Image height too small to split effectively. Using full image features for both halves.")
             full_features = extract_single_feature_from_image(pil_image, feature_extractor, model, device)
             return full_features, full_features # Return same features for upper/lower

        mid_point = height // 2

        # Crop upper and lower halves
        # Image.crop takes a 4-tuple (left, upper, right, lower)
        upper_img = pil_image.crop((0, 0, width, mid_point))
        lower_img = pil_image.crop((0, mid_point, width, height))

        # Extract features for both halves
        # Note: ViTFeatureExtractor handles resizing internally if needed,
        # but applying it to already cropped small parts might be suboptimal.
        # The default behavior resizes based on 'shortest edge'.
        # Consider resizing the *whole* image first if results are poor.
        features_upper = extract_single_feature_from_image(upper_img, feature_extractor, model, device)
        features_lower = extract_single_feature_from_image(lower_img, feature_extractor, model, device)

        if features_upper is None or features_lower is None:
            return None # Propagate error if feature extraction failed for either part

        return features_upper, features_lower # Return as a tuple

    except Exception as e:
        st.error(f"Error splitting/processing image for feature extraction: {e}")
        return None


# --- Data Persistence Functions ---
def get_next_person_id():
    """Gets the next available person ID."""
    # (Function remains the same as before)
    try:
        if os.path.exists(PERSON_ID_COUNTER_FILE):
            with open(PERSON_ID_COUNTER_FILE, 'r') as f:
                person_id = int(f.read().strip())
        else:
            person_id = 0
        next_person_id = person_id + 1
        with open(PERSON_ID_COUNTER_FILE, 'w') as f:
            f.write(str(next_person_id))
        return person_id # Return the current highest ID to be used now
    except Exception as e:
        st.warning(f"Could not read/write person ID counter: {e}. Using session state fallback.")
        if 'person_id_counter' not in st.session_state:
             st.session_state.person_id_counter = 0 # Start with 0
        else:
             st.session_state.person_id_counter += 1
        # We don't write back to file in this fallback case easily without knowing the next ID
        return st.session_state.person_id_counter

def save_image(image_bytes, person_id, image_index):
    """Saves image bytes to a structured directory and returns the path."""
    # (Function remains the same as before)
    person_dir = os.path.join(IMAGE_DIR, f"person_{person_id}")
    os.makedirs(person_dir, exist_ok=True)
    timestamp = int(time.time() * 1000) # Add timestamp for uniqueness
    filename = f"img_{image_index}_{timestamp}.jpg"
    filepath = os.path.join(person_dir, filename)
    try:
        with open(filepath, "wb") as f:
            f.write(image_bytes)
        return filepath
    except Exception as e:
        st.error(f"Error saving image {filepath}: {e}")
        return None

def load_persistent_data():
    """Loads features, image paths, and cluster model from disk."""
    # (Function remains the same as before, but interpretation of features changes)
    features = None
    image_paths = []
    kmeans_model = None
    if os.path.exists(FEATURES_FILE) and os.path.exists(IMAGE_PATHS_FILE):
        try:
            if os.path.getsize(FEATURES_FILE) > 0 and os.path.getsize(IMAGE_PATHS_FILE) > 0:
                 features = np.load(FEATURES_FILE)
                 # --- Add check for expected feature dimension ---
                 expected_dim = VIT_EMBEDDING_DIM * 2
                 if features.shape[1] != expected_dim:
                     st.error(f"Loaded features have unexpected dimension ({features.shape[1]}). Expected {expected_dim} (split features). Please Reset ALL Data.")
                     # Invalidate loaded data
                     features = None
                     image_paths = []
                     # Optionally remove corrupted files
                     # os.remove(FEATURES_FILE)
                     # os.remove(IMAGE_PATHS_FILE)
                 else:
                     with open(IMAGE_PATHS_FILE, 'rb') as f:
                         image_paths = pickle.load(f)
                     st.success(f"Loaded {len(image_paths)} images and {features.shape[0]} split features from disk.")
            else:
                st.warning("Feature or path file exists but is empty. Treating as no data.")
                if os.path.exists(FEATURES_FILE): os.remove(FEATURES_FILE)
                if os.path.exists(IMAGE_PATHS_FILE): os.remove(IMAGE_PATHS_FILE)

        except Exception as e:
            st.error(f"Error loading features/paths: {e}")
            if os.path.exists(FEATURES_FILE): os.remove(FEATURES_FILE)
            if os.path.exists(IMAGE_PATHS_FILE): os.remove(IMAGE_PATHS_FILE)

    if os.path.exists(CLUSTER_MODEL_FILE):
        try:
             if os.path.getsize(CLUSTER_MODEL_FILE) > 0:
                with open(CLUSTER_MODEL_FILE, 'rb') as f:
                    kmeans_model = pickle.load(f)
                # --- Add check for expected centroid dimension ---
                if kmeans_model and hasattr(kmeans_model, 'cluster_centers_'):
                     expected_centroid_dim = VIT_EMBEDDING_DIM * 2
                     if kmeans_model.cluster_centers_.shape[1] != expected_centroid_dim:
                          st.error(f"Loaded KMeans centroids have unexpected dimension ({kmeans_model.cluster_centers_.shape[1]}). Expected {expected_centroid_dim}. Please Reset ALL Data or Retrain Clusters.")
                          kmeans_model = None # Invalidate model
                          # Optionally remove corrupted file
                          # os.remove(CLUSTER_MODEL_FILE)
                     else:
                          st.success("Loaded existing cluster model.")
                elif kmeans_model:
                     st.warning("Loaded object seems to be a KMeans model but lacks cluster centers.")
                     kmeans_model = None # Invalidate
             else:
                  st.warning("Cluster model file exists but is empty.")
                  if os.path.exists(CLUSTER_MODEL_FILE): os.remove(CLUSTER_MODEL_FILE)

        except Exception as e:
            st.error(f"Error loading KMeans model: {e}")
            if os.path.exists(CLUSTER_MODEL_FILE): os.remove(CLUSTER_MODEL_FILE)

    return features, image_paths, kmeans_model

def save_persistent_data(features, image_paths, kmeans_model):
    """Saves features, image paths, and cluster model to disk."""
    # (Function remains the same as before)
    try:
        np.save(FEATURES_FILE, features)
        with open(IMAGE_PATHS_FILE, 'wb') as f:
            pickle.dump(image_paths, f)
        if kmeans_model:
            with open(CLUSTER_MODEL_FILE, 'wb') as f:
                pickle.dump(kmeans_model, f)
        print("Persistent data saved successfully.") # Debugging
    except Exception as e:
        st.error(f"Error saving persistent data: {e}")

# --- Clustering Function ---
def update_clusters(all_features):
    """Trains or updates the KMeans model on the combined features."""
    # (Function remains largely the same, but operates on double-dim features)
    if all_features is None or len(all_features) < 1: # Need at least 1 sample
        st.info("Not enough data points to perform clustering.")
        return None

    # Determine the number of people (clusters)
    # This estimation is still approximate
    num_people_estimate = max(1, len(all_features) // NUM_INITIAL_PHOTOS)
    # Ensure n_clusters is not more than n_samples
    num_clusters = min(num_people_estimate, len(all_features))

    if num_clusters < 1:
         st.warning("Cannot determine a valid number of clusters (must be >= 1).")
         return None

    st.write(f"Attempting to train KMeans with {num_clusters} clusters on {len(all_features)} samples (dim={all_features.shape[1]})...")
    try:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        kmeans.fit(all_features)
        st.success(f"KMeans clustering successful with {num_clusters} clusters.")
        return kmeans
    except ValueError as ve:
         st.error(f"Error during KMeans clustering: {ve}")
         st.error(f"Details: n_samples={len(all_features)}, n_clusters={num_clusters}")
         return None
    except Exception as e:
        st.error(f"An unexpected error occurred during KMeans clustering: {e}")
        return None

# --- Initialize Model (after function definitions) ---
feature_extractor, model, device = load_model()

# --- Streamlit App Logic ---
st.title("Person Re-ID with Split Features & Clustering (ViT)")

# --- State Management & Data Loading ---
if 'step' not in st.session_state:
    st.session_state.step = 'init'
    st.session_state.all_features = None # Will store concatenated (upper+lower) features
    st.session_state.all_image_paths = []
    st.session_state.kmeans_model = None
    st.session_state.query_image_bytes = None
    st.session_state.query_features_upper = None # Store split query features
    st.session_state.query_features_lower = None
    st.session_state.new_person_images_bytes = []
    st.session_state.new_person_features_combined = [] # Store combined features for new person
    st.session_state.new_person_image_paths = []
    st.session_state.current_person_id = None
    st.session_state.match_cluster_id = None
    st.session_state.match_score = None
    st.session_state.data_loaded = False

# Load persistent data once
if not st.session_state.data_loaded:
    print("Attempting to load persistent data...")
    with st.spinner("Loading existing data..."):
        features, paths, model_loaded = load_persistent_data()
    if features is not None and paths:
        st.session_state.all_features = features
        st.session_state.all_image_paths = paths
        print(f"Loaded features shape: {st.session_state.all_features.shape}") # Should be N x (DIM*2)
    if model_loaded:
        st.session_state.kmeans_model = model_loaded
        if hasattr(model_loaded, 'cluster_centers_'):
             print(f"Loaded KMeans model with {st.session_state.kmeans_model.n_clusters} clusters. Centroid dim: {st.session_state.kmeans_model.cluster_centers_.shape[1]}") # Should be DIM*2
    st.session_state.data_loaded = True
    st.session_state.step = 'check_identity'
    st.rerun()

# --- Workflow Steps ---

# == Step 0: Initialization / Loading ==
if st.session_state.step == 'init':
    st.info("Initializing and loading data...")

# == Step 1: Check Identity ==
elif st.session_state.step == 'check_identity':
    st.header("Step 1: Identify Yourself")
    st.write("Take a photo. We'll check if you're already in our database using split features.")

    img_file_buffer_query = st.camera_input("Take Query Photo", key="cam_query_identity")

    if img_file_buffer_query is not None:
        img_bytes = img_file_buffer_query.getvalue()
        with st.spinner("Processing query image (splitting & extracting features)..."):
            # Extract split features
            features_split = extract_features_transformer_split(img_bytes, feature_extractor, model, device)

        if features_split is not None:
            st.session_state.query_image_bytes = img_bytes
            st.session_state.query_features_upper, st.session_state.query_features_lower = features_split
            st.image(img_bytes, caption="Query Photo", width=200)

            # Check against existing clusters
            if st.session_state.kmeans_model is not None and \
               st.session_state.all_features is not None and \
               len(st.session_state.all_features) > 0 and \
               hasattr(st.session_state.kmeans_model, 'cluster_centers_'):

                centroids_combined = st.session_state.kmeans_model.cluster_centers_
                num_clusters = centroids_combined.shape[0]
                expected_dim = VIT_EMBEDDING_DIM # Dimension of *one* part (upper or lower)

                # Check centroid dimensions before splitting
                if centroids_combined.shape[1] != expected_dim * 2:
                     st.error(f"Centroid dimension mismatch ({centroids_combined.shape[1]} vs expected {expected_dim*2}). Cannot perform matching. Please reset data or retrain clusters.")
                else:
                    try:
                        # Split centroids into upper and lower parts
                        centroids_upper = centroids_combined[:, :expected_dim]
                        centroids_lower = centroids_combined[:, expected_dim:]

                        # Calculate similarities separately
                        similarities_upper = cosine_similarity(st.session_state.query_features_upper, centroids_upper)[0]
                        similarities_lower = cosine_similarity(st.session_state.query_features_lower, centroids_lower)[0]

                        # --- Combine Similarities ---
                        # Option 1: Average
                        combined_similarities = (similarities_upper + similarities_lower) / 2.0
                        # Option 2: Max (take the score of the better matching half)
                        # combined_similarities = np.maximum(similarities_upper, similarities_lower)
                        # Option 3: Weighted average (e.g., if upper body is usually more discriminative)
                        # combined_similarities = 0.6 * similarities_upper + 0.4 * similarities_lower

                        best_match_idx = np.argmax(combined_similarities)
                        st.session_state.match_score = combined_similarities[best_match_idx]

                        print(f"Upper Similarities: {similarities_upper}") # Debug
                        print(f"Lower Similarities: {similarities_lower}") # Debug
                        print(f"Combined Similarities: {combined_similarities}") # Debug
                        print(f"Best match score (Combined): {st.session_state.match_score}, Cluster Index: {best_match_idx}")

                        # Decision based on combined threshold
                        if st.session_state.match_score >= MATCH_THRESHOLD:
                            st.session_state.match_cluster_id = best_match_idx
                            st.success(f"Match found! You seem to be Person Cluster #{best_match_idx+1} (Combined Score: {st.session_state.match_score:.4f}).")
                            st.session_state.step = 'show_results_existing'
                            st.rerun()
                        else:
                            st.warning(f"No close match found (Highest Combined Score: {st.session_state.match_score:.4f}). Let's add you as a new person.")
                            # Start the 'add new person' flow, using the query photo's data
                            st.session_state.new_person_images_bytes = [st.session_state.query_image_bytes]
                            # Combine the extracted features for the first image
                            combined_query_features = np.concatenate((st.session_state.query_features_upper, st.session_state.query_features_lower), axis=1)
                            st.session_state.new_person_features_combined = [combined_query_features]
                            st.session_state.step = 'capture_initial'
                            st.rerun()
                    except ValueError as e_sim:
                         st.error(f"Error during similarity calculation: {e_sim}. Check feature dimensions.")
                         print(f"Query upper shape: {st.session_state.query_features_upper.shape}, Query lower shape: {st.session_state.query_features_lower.shape}")
                         print(f"Centroids combined shape: {centroids_combined.shape}")
                         st.session_state.step = 'check_identity' # Allow retry
                         st.rerun()

            else:
                st.info("Database is empty or no valid clusters found. Let's add you as the first person.")
                # Start 'add new person' flow
                st.session_state.new_person_images_bytes = [st.session_state.query_image_bytes]
                # Combine the extracted features for the first image
                combined_query_features = np.concatenate((st.session_state.query_features_upper, st.session_state.query_features_lower), axis=1)
                st.session_state.new_person_features_combined = [combined_query_features]
                st.session_state.step = 'capture_initial'
                st.rerun()
        else:
            st.error("Could not process the query photo (feature extraction failed). Please try again.")


# == Step 2: Capture Initial Photos (for New Person) ==
elif st.session_state.step == 'capture_initial':
    st.header(f"Step 2: Add New Person - Capture {NUM_INITIAL_PHOTOS} Photos")
    st.write(f"We need {NUM_INITIAL_PHOTOS} photos from different angles.")
    photos_taken = len(st.session_state.new_person_images_bytes)
    st.write(f"Photo {photos_taken}/{NUM_INITIAL_PHOTOS} captured.")

    # Ensure we started correctly (should have 1 photo/feature from query)
    if photos_taken == 0 or len(st.session_state.new_person_features_combined) == 0:
         st.warning("Started 'Add New Person' incorrectly. Please go back and take a query photo first.")
         if st.button("Go Back to Identify Step"):
              # Clear potentially inconsistent state
              st.session_state.new_person_images_bytes = []
              st.session_state.new_person_features_combined = []
              st.session_state.step = 'check_identity'
              st.rerun()

    # Display captured images
    if st.session_state.new_person_images_bytes:
        st.write("Captured Photos for New Person:")
        cols = st.columns(NUM_INITIAL_PHOTOS)
        for i, img_bytes in enumerate(st.session_state.new_person_images_bytes):
             cols[i % NUM_INITIAL_PHOTOS].image(img_bytes, caption=f"New Person Photo {i+1}", width=150)

    # Camera input for next photos
    if len(st.session_state.new_person_images_bytes) < NUM_INITIAL_PHOTOS:
        next_photo_num = len(st.session_state.new_person_images_bytes) + 1
        img_file_buffer = st.camera_input(f"Take Initial Photo #{next_photo_num}", key=f"cam_initial_{next_photo_num}")

        if img_file_buffer is not None:
            img_bytes = img_file_buffer.getvalue()
            with st.spinner(f"Processing photo {next_photo_num} (splitting & extracting features)..."):
                 features_split = extract_features_transformer_split(img_bytes, feature_extractor, model, device)

            if features_split is not None:
                features_upper, features_lower = features_split
                # Combine features before storing
                combined_features = np.concatenate((features_upper, features_lower), axis=1)

                st.session_state.new_person_images_bytes.append(img_bytes)
                st.session_state.new_person_features_combined.append(combined_features)
                st.success(f"Photo {next_photo_num} captured and features processed.")
                st.rerun()
            else:
                st.warning(f"Could not extract features from Photo {next_photo_num}. Please try again.")

    # Process adding the new person
    if len(st.session_state.new_person_images_bytes) == NUM_INITIAL_PHOTOS and len(st.session_state.new_person_features_combined) == NUM_INITIAL_PHOTOS:
        st.success(f"All {NUM_INITIAL_PHOTOS} photos captured. Adding to database...")

        with st.spinner("Saving images and updating database..."):
            st.session_state.current_person_id = get_next_person_id()
            st.write(f"Assigning Person ID: {st.session_state.current_person_id}")

            st.session_state.new_person_image_paths = []
            save_success = True
            for i, img_bytes in enumerate(st.session_state.new_person_images_bytes):
                img_path = save_image(img_bytes, st.session_state.current_person_id, i)
                if img_path:
                    st.session_state.new_person_image_paths.append(img_path)
                else:
                    st.error(f"Failed to save image {i+1}. Aborting add.")
                    save_success = False
                    break

            if save_success and len(st.session_state.new_person_image_paths) == NUM_INITIAL_PHOTOS:
                # Concatenate all *new* combined features for this person
                new_features_to_add = np.concatenate(st.session_state.new_person_features_combined, axis=0)

                # Append to global features
                if st.session_state.all_features is None or len(st.session_state.all_features) == 0:
                    st.session_state.all_features = new_features_to_add
                    st.session_state.all_image_paths = st.session_state.new_person_image_paths
                else:
                    # --- Check dimension compatibility before concatenating ---
                    if st.session_state.all_features.shape[1] == new_features_to_add.shape[1]:
                         st.session_state.all_features = np.concatenate([st.session_state.all_features, new_features_to_add], axis=0)
                         st.session_state.all_image_paths.extend(st.session_state.new_person_image_paths)
                    else:
                         st.error(f"Cannot add new features. Dimension mismatch: Existing({st.session_state.all_features.shape[1]}) vs New({new_features_to_add.shape[1]}). Please Reset ALL Data.")
                         # Prevent inconsistent state - rollback add attempt
                         st.session_state.step = 'check_identity'
                         st.session_state.new_person_images_bytes = []
                         st.session_state.new_person_features_combined = []
                         st.session_state.new_person_image_paths = []
                         st.rerun()


                # Update and save clusters (only if concatenation succeeded)
                if st.session_state.all_features is not None: # Check again after potential concat error
                    st.session_state.kmeans_model = update_clusters(st.session_state.all_features)
                    save_persistent_data(st.session_state.all_features, st.session_state.all_image_paths, st.session_state.kmeans_model)
                    st.success(f"New Person (ID: {st.session_state.current_person_id}) added successfully!")
                    st.session_state.step = 'show_results_new'
                    st.rerun()

            else:
                 st.error("Failed to save all images. Cannot add new person.")
                 # Reset the attempt
                 st.session_state.step = 'check_identity'
                 st.session_state.new_person_images_bytes = []
                 st.session_state.new_person_features_combined = []
                 st.session_state.new_person_image_paths = []
                 st.rerun()

# == Step 3a: Show Results (Existing Person Found) ==
elif st.session_state.step == 'show_results_existing':
    st.header("Step 3: Re-Identification Result - Match Found")

    st.subheader("Query Image:")
    if st.session_state.query_image_bytes:
        st.image(st.session_state.query_image_bytes, width=300)

    st.metric(label=f"Matched Cluster (Person Estimate)", value=f"#{st.session_state.match_cluster_id + 1}")
    # Be clear this is the combined score
    st.metric(label="Confidence (Avg. Cosine Similarity - Upper/Lower)", value=f"{st.session_state.match_score:.4f}")

    st.subheader("Gallery Images from Matched Cluster:")
    # (This part remains the same - displays images associated with the cluster label)
    if st.session_state.kmeans_model is not None and st.session_state.all_image_paths and hasattr(st.session_state.kmeans_model, 'labels_'):
        try:
            labels = st.session_state.kmeans_model.labels_
            if st.session_state.match_cluster_id >= len(np.unique(labels)):
                 st.warning(f"Match Cluster ID {st.session_state.match_cluster_id} invalid for current labels {np.unique(labels)}. Showing all images.")
                 matched_indices = list(range(len(st.session_state.all_image_paths)))
            else:
                 matched_indices = [i for i, label in enumerate(labels) if label == st.session_state.match_cluster_id]

            if matched_indices:
                 valid_indices = [idx for idx in matched_indices if idx < len(st.session_state.all_image_paths)]
                 if not valid_indices:
                     st.warning("Indices found for cluster, but they are out of bounds for the image path list.")
                 else:
                    max_display = min(5, len(valid_indices))
                    display_indices = valid_indices[:max_display]
                    cols = st.columns(max_display)
                    for i, img_idx in enumerate(display_indices):
                        img_path = st.session_state.all_image_paths[img_idx]
                        if os.path.exists(img_path):
                            try:
                                cols[i].image(img_path, caption=f"Cluster Img {i+1}\n(Orig Idx {img_idx})", width=150)
                            except Exception as e_img:
                                cols[i].error(f"Error loading {os.path.basename(img_path)}: {e_img}")
                        else:
                            cols[i].warning(f"Img not found:\n{os.path.basename(img_path)}")
            else:
                st.warning(f"No images found associated with cluster label {st.session_state.match_cluster_id} in the current data.")
        except IndexError as e_idx:
             st.error(f"Error accessing cluster labels or image paths: {e_idx}. Data might be inconsistent.")
        except Exception as e:
             st.error(f"An error occurred displaying cluster images: {e}")
    else:
        st.warning("Cannot display cluster images - model, labels, or image paths are missing/invalid.")


# == Step 3b: Show Results (New Person Added) ==
elif st.session_state.step == 'show_results_new':
    st.header("Step 3: Re-Identification Result - New Person Added")
    # (This part remains the same - displays the images just added)
    st.success(f"You have been added to the database as Person ID: {st.session_state.current_person_id}")
    st.write("These are the images we saved for you:")

    if not st.session_state.new_person_images_bytes:
         st.warning("No images found in the session state for the newly added person.")
    else:
        cols = st.columns(min(NUM_INITIAL_PHOTOS, len(st.session_state.new_person_images_bytes)))
        for i, img_bytes in enumerate(st.session_state.new_person_images_bytes):
            caption_text = f"Saved Photo {i+1}"
            display_img = img_bytes
            if i < len(st.session_state.new_person_image_paths):
                img_path = st.session_state.new_person_image_paths[i]
                caption_text = f"Saved Photo {i+1}\n({os.path.basename(img_path)})"
                try:
                    if os.path.exists(img_path):
                        display_img = Image.open(img_path)
                    else:
                         st.warning(f"Image file not found: {img_path}. Displaying captured bytes.")
                         display_img = img_bytes
                except Exception as e_open:
                    st.warning(f"Could not open saved image {img_path}: {e_open}. Displaying captured bytes.")
                    display_img = img_bytes

            cols[i % len(cols)].image(display_img, caption=caption_text, width=200)


# --- Control Buttons ---
st.divider()
col1, col2 = st.columns([0.7, 0.3])

with col1:
    if st.session_state.step != 'check_identity':
        if st.button("Check Another Person / Start Over", key="start_over_button"):
            st.session_state.step = 'check_identity'
            st.session_state.query_image_bytes = None
            st.session_state.query_features_upper = None # Clear split query features
            st.session_state.query_features_lower = None
            st.session_state.new_person_images_bytes = []
            st.session_state.new_person_features_combined = [] # Clear combined new features
            st.session_state.new_person_image_paths = []
            st.session_state.current_person_id = None
            st.session_state.match_cluster_id = None
            st.session_state.match_score = None
            st.rerun()

with col2:
     if st.button("⚠️ Reset ALL Data", type="primary", help="Deletes all saved images, features, and clusters!", key="reset_all_button"):
        st.session_state.confirm_delete = True

     if 'confirm_delete' in st.session_state and st.session_state.confirm_delete:
        confirm = st.checkbox("Confirm deletion of all persistent data", key="confirm_delete_checkbox")
        if confirm:
            # (Deletion logic remains the same)
            if os.path.exists(FEATURES_FILE): os.remove(FEATURES_FILE)
            if os.path.exists(IMAGE_PATHS_FILE): os.remove(IMAGE_PATHS_FILE)
            if os.path.exists(CLUSTER_MODEL_FILE): os.remove(CLUSTER_MODEL_FILE)
            if os.path.exists(PERSON_ID_COUNTER_FILE): os.remove(PERSON_ID_COUNTER_FILE)
            if os.path.exists(IMAGE_DIR): shutil.rmtree(IMAGE_DIR); os.makedirs(IMAGE_DIR, exist_ok=True)
            for key in list(st.session_state.keys()): del st.session_state[key]
            st.cache_resource.clear()
            st.cache_data.clear()
            st.success("All persistent data has been deleted. Restarting app...")
            # Don't need to delete confirm_delete flag here, it's cleared by clearing all state keys
            time.sleep(2)
            st.rerun()
        if st.button("Cancel Deletion", key="cancel_delete_button"):
             st.session_state.confirm_delete = False
             st.rerun()


# --- Display DB Info (Sidebar) ---
st.sidebar.title("Database Info (Split Features)")
data_available = st.session_state.all_features is not None and len(st.session_state.all_image_paths) > 0

if data_available:
    st.sidebar.metric("Total Images Stored", len(st.session_state.all_image_paths))
    st.sidebar.metric("Total Features Stored", len(st.session_state.all_features))
    st.sidebar.write(f"Feature Dimension: {st.session_state.all_features.shape[1]} ({VIT_EMBEDDING_DIM} Upper + {VIT_EMBEDDING_DIM} Lower)") # Show dimension
    if st.session_state.kmeans_model and hasattr(st.session_state.kmeans_model, 'n_clusters'):
         st.sidebar.metric("Number of Clusters (People)", st.session_state.kmeans_model.n_clusters)
    else:
         st.sidebar.write("No cluster model found or model is invalid.")
else:
    st.sidebar.write("No data loaded or stored yet.")

if data_available:
    if st.sidebar.button("Force Retrain Clusters"):
        if st.session_state.all_features is not None and len(st.session_state.all_features) > 0:
            with st.spinner("Retraining clusters..."):
                st.session_state.kmeans_model = update_clusters(st.session_state.all_features)
            if st.session_state.kmeans_model:
                save_persistent_data(st.session_state.all_features, st.session_state.all_image_paths, st.session_state.kmeans_model)
                st.sidebar.success("Clusters retrained and saved.")
                st.rerun()
            else:
                 st.sidebar.error("Cluster retraining failed. Check logs.")
        else:
            st.sidebar.warning("No features available to train clusters.")