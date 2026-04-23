# Streamlit Apps

ASCOTA ships with a set of **Streamlit-based demo applications**.  
These apps provide a simple UI for archaeologists and researchers to test the core
functions interactively — no coding required.

Each app wraps functions from `ascota_core` to demonstrate the **Phase 1 pipeline**
(capabilities for segmentation, scale estimation, and color correction).

---

## How to run

From the project root:

```bash
# Activate environment first
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Then run any of the apps:
streamlit run tests/streamlits/color_correct_streamlit.py
streamlit run tests/streamlits/scale_streamlit.py
streamlit run tests/streamlits/segment_streamlit.py
streamlit run tests/streamlits/color_clustering_streamlit.py
streamlit run tests/streamlits/color_clustering_lab_streamlit.py
streamlit run tests/streamlits/decoration_classification_streamlit.py
streamlit run tests/streamlits/type_classification_streamlit.py
streamlit run tests/streamlits/type_pottery_non_pottery_streamlit.py
streamlit run tests/streamlits/texture_clustering_streamlit.py
```

Open your browser at [http://localhost:8501](http://localhost:8501).

---

## Available Apps

### 🎨 Color Correction

**File:** `color_correct_streamlit.py`

* Detects color reference cards in input images.
* Applies **color correction** to normalize pottery sherd images to a selected target image.
* Useful for ensuring consistent analysis across lighting conditions and excavation seasons.
* Allows testing of various color correction/transformation algorithms.

---

### 🌈 Color Clustering

**File:** `color_clustering_streamlit.py`
* Clusters images based on **dominant color features**.
* Groups photos taken under similar lighting conditions.
* Helps archaeologists quickly identify and organize images by color similarity and lighting.
* Useful for batch processing and allows us to select representative images for color correction.

---

### 📏 Scale Estimation

**File:** `scale_streamlit.py`

* Detects measurement cards in excavation images.
* Computes a **pixels-per-centimeter ratio**.
* Estimates the **surface area of sherds/pottery pieces** from photos.
* Provides archaeologists with approximate real-world measurements directly from images.

---

### ✂️ Segmentation

**File:** `segment_swatch_streamlit.py`

* Runs **segmentation models** (RMBG-1.4 + OpenCV) to isolate sherds and measurement cards.
* Produces clean masks and regions of interest (ROIs) for further analysis.
* Segments & Classifies color cards, measurement cards in the image.
* Generates an image swatch of the sherd.

---

### 🖌️ Color Clustering (Lab)

**File:** `color_clustering_lab_streamlit.py`

* Runs Lab-summary-based clustering experiments for grouped pottery imagery.
* Helps evaluate color-separation behavior before running full product workflows.

---

### 🧩 Decoration Classification

**File:** `decoration_classification_streamlit.py`

* Demonstrates decoration classification over segmented sherd images.
* Useful for testing model outputs and confidence behavior quickly.

---

### 🏺 Type Classification

**File:** `type_classification_streamlit.py`

* Demonstrates multi-stage pottery type classification.
* Useful for validating labels and confidence distributions on sample sets.

---

### 🏷️ Pottery Gate

**File:** `type_pottery_non_pottery_streamlit.py`

* Runs pottery vs non-pottery gating for quick dataset screening.
* Useful as a pre-step before detailed type classification.

---

### 🪨 Texture Clustering

**File:** `texture_clustering_streamlit.py`

* Groups images by texture similarity using extracted surface features.
* Helps inspect texture-driven grouping before downstream labeling.

---

## Next Steps

These apps serve as **experimental prototypes** for Phase 1 of the ASCOTA pipeline.
In later phases, additional apps will demonstrate:

* **Classification** (Phase 2): automatic labeling by type, color, texture, and decoration/pattern.
* **Full Workflow** (Phase 3): an end-to-end UI for archaeologists to process excavation photos.