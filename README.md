# PathSnap v2 — From-Scratch High-Accuracy Pipeline

End-to-end leak-free pipeline built specifically for the real PathSnap dataset scale (80 training trips / ~208k GPS points / up to 307k road arcs per trip, and 10 student-testing trips).

## Pipeline Stages
1. **Preprocessing & Feature Engineering** (`src/preprocessing.py`)
   - Validates/cleans raw `.track`, `.nodes`, `.arcs`, `.route` files.
   - Projects each trip dynamically to its own **UTM EPSG zone** (meters) to handle global trip locations (`UTM 32614`–`32735`).
   - Pre-filters road arcs by a leak-free `350 m` bounding box around the track, indexes them via a vectorized densified `cKDTree`, retrieves top `K = 8` candidates within `120 m`, and logs **Candidate Miss Rate**.
   - Aligns `.track` to `.route` via vectorized **Monotonic Dynamic Programming** (`np.minimum.accumulate`), handling `1.26x–57x` observation-to-arc ratios and loops.
   - Engineers **37 leak-free candidate features**: spatial proximity, within-observation relative ranks/z-scores/Gaussian weights, directional alignment (to separate opposing two-way arcs), `progress_along_arc` ($u_{t+1}-u_{t-1}$), multi-step smoothed distances, gap-aware kinematics, node degrees, and leak-free predecessor/successor topological candidate-chain connectivity.
2. **Leak-Free Feature Selection** (`src/feature_selection.py`)
   - Fit strictly on the **80 training trips**.
   - Drops zero-variance features and collinear duplicates (`|r| > 0.97`, keeping higher Mutual Information).
   - Ranks remaining features by consensus of **Mutual Information + Tree Gain + Permutation Importance**.
   - Runs **GroupKFold CV subset ablation** to pick the highest-accuracy, cleanest feature subset.
3. **Training & Comparing the Best 5+ Models** (`src/train_models.py`)
   - `LogisticRegression` (*Linear & Probabilistic*)
   - `RandomForest` (*Bagging Ensemble*)
   - `ExtraTrees` (*Tree-based / Randomized Ensemble*)
   - `LightGBM` (*Boosting — Leaf-wise GBDT*, fallback `HistGradientBoosting`)
   - `MLPNeuralNet` (*Neural Networks*, with authentic trip-grouped epoch-wise loss curves)
   - `XGBoost` (*Boosting — Depth-wise GBDT*, automatically included when installed)
   - Uses natural class priors (calibrated probabilities for low Log Loss & clean HMM emissions) and 5-fold **`GroupKFold` grouped by `trip_id`**.

## Run Everything in One Command
```powershell
cd C:\Users\araya\.gemini\antigravity\scratch\pathsnap_v2
pip install -r requirements.txt
python run_pipeline.py
```
To run individual stages after Stage 1 caches the feature tables:
```powershell
python src/preprocessing.py
python src/feature_selection.py
python src/train_models.py
```
