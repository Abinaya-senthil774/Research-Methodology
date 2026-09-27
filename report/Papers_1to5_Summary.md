# Comparative Summary of Papers 1–5: Pest/RPW Detection Using Deep Learning

---

## Paper 1 — AMS-YOLO: Multi-Scale Feature Integration for Intelligent Plant Protection Against Maize Pests
*Deng, Fang, Ullah, Hou & Yu — Frontiers in Plant Science, 2025*

**Problem Statement**
Maize suffers yield losses of up to 22.5% due to pests. Existing detection models perform poorly because of (a) dramatic morphological changes across a pest's developmental stages, (b) high visual similarity between different pest species/orders, and (c) complex field backgrounds (foliage, soil, lighting/shadow variation) that reduce detection precision.

**Objectives**
- Build a lightweight, real-time detector that reliably distinguishes maize pests across life stages and species.
- Improve feature discrimination in cluttered field backgrounds without increasing compute cost.
- Validate deployability on resource-constrained edge hardware.

**Research Gaps Identified**
1. Datasets/models usually cover a single developmental stage or environment, limiting generalization to morphological change.
2. Poor discrimination between pests with high inter-class similarity.
3. Poor trade-off between accuracy and computational cost for edge deployment.

**Gaps Covered by This Paper**
- Introduces **SMCA** (Spatial-Multiscale Context Attention) for background-robust, similarity-aware feature extraction.
- Introduces **MSBlock** for multi-scale feature fusion across developmental stages.
- Introduces **AMConv** (Average-Max pooling Convolution), a dual-path downsampling module that preserves fine detail, reducing parameter count.

**Methodology**
- Base model: **YOLOv8n**, enhanced with three modules (SMCA in backbone C2f, MSBlock in neck at P3/P4/P5, AMConv replacing standard downsampling).
- SMCA combines multi-head spatial self-attention with local+global channel-context branches (adaptive 1D convolution), fused via a learnable weighted sigmoid gate.
- MSBlock splits expanded channels into groups processed with 1×1/3×3/5×5 kernels in a cumulative (residual-chained) fashion.
- AMConv splits channels into an average-pooled path (stride-2 conv) and a max-pooled + pointwise-conv path, then concatenates.

**Setup**
- Dataset: 13 maize pest species from **IP102** (4,293 valid images) + 242 self-collected field images (Jilin Agricultural University, July 2024), split 8:1:1 (train/val/test).
- Augmentation: vertical flip, brightness, Gaussian blur, motion blur, popcorn noise.
- Final dataset after augmentation/cleaning: 9,371 images.
- Hardware: NVIDIA RTX 4070 SUPER, Intel i5-13600KF, 64 GB RAM, PyTorch 1.12.1, CUDA 11.3.
- Training: 200 epochs, image size 640×640, batch 16, optimizer AdamW, LR 0.002, momentum 0.937, weight decay 0.0005.
- Edge deployment test: NVIDIA Jetson Nano.

**Metrics**
Precision, Recall, mAP50, mAP50:95, Parameter count, Model weight (MB), inference/FPS, CPU utilization.

**Results**
- AMS-YOLO vs. baseline YOLOv8n: **Precision 90.0% (+3.1%)**, **Recall 89.8% (+3.7%)**, **mAP50 94.2% (+3.2%)**, **mAP50:95 73.7% (+4.0%)**.
- Model size only **5.3 MB** (−15.9% vs YOLOv8n), parameters −19.6%, compute −16%.
- Outperformed SSD, RetinaNet, RT-DETR, and YOLOv3/v5/v6/v9/v10/v11/v12 variants on mAP50/mAP50:95 while using far fewer parameters than RT-DETR (2.53M vs 42.8M).
- Ablation: SMCA alone raised Precision 86.9%→89.2%; MSBlock alone raised Recall to 87.1%; AMConv alone cut parameters to 2.84M; all three combined gave the best overall result (Model 8).
- Jetson Nano: inference time 89.5 ms→69.4 ms, FPS 10.52→12.25 (stable phase), CPU utilization 74.7%→58.5%.
- Confusion-matrix analysis: black cutworm accuracy 78%→81%; cutworm confusion 17%→11%; aphid misclassification 11%→7%.

---

## Paper 2 — Maize-YOLO: A New High-Precision and Real-Time Method for Maize Pest Detection
*Yang, Xing, Wang, Dong, Gao, Liu, Zhang, Li & Zhao — Insects, 2023*

**Problem Statement**
Maize pest detection needs to be fast, accurate, and computationally light for real-time field use, but existing YOLO-family detectors cannot simultaneously balance accuracy, speed, and computational cost.

**Objectives**
- Propose a maize-pest-specific detector that improves accuracy and speed while reducing FLOPs relative to YOLOv7.
- Evaluate on a challenging subset of the large-scale IP102 pest dataset.

**Research Gaps Identified**
- Few studies apply YOLOv7 to plant pest detection.
- Existing DL pest detectors rarely achieve a good three-way balance of accuracy, computational effort, and detection speed simultaneously.
- Limited work specifically targeting maize pests (vs. general crop pests).

**Gaps Covered**
- Inserts **CSPResNeXt-50** (replacing some ELAN blocks) to cut memory-access cost/parameters while raising accuracy.
- Replaces ELAN-W with **VoVGSCSP** (built on lightweight GSConv) in the neck to raise inference speed without hurting accuracy.

**Methodology**
- Base architecture: YOLOv7 (Input → Backbone → Head), with Backbone = ELAN + CBS + CSPResNeXt-50, Head/Neck aggregation via SPPCSPC, ELAN-W→VoVGSCSP, and REP modules for final channel adjustment.
- CSPResNeXt-50: CSPNet-style split-path ResNeXt block, halving channels through ResUnit to lower MAC.
- VoVGSCSP: GS-bottleneck (GSConv ×2) + one-shot aggregation in a cross-stage partial design.

**Setup**
- Dataset: **IP13**, a 13-class subset of IP102 curated for maize-damaging pests (4,533 images total; e.g., Mole cricket 868, Aphids 874, White margined moth 49).
- 5-fold cross-validation: ~80% train (3,600 imgs)/20% test (932 imgs) per fold, averaged.
- Input resized to 640×640 (no other pre-processing to preserve originality).
- Hardware: 16× Intel Xeon Gold 5218 CPU + 2× NVIDIA RTX 3090, PyTorch 1.11.0, Ubuntu 20.04, Python 3.9.
- Training: epoch 300, batch 32, Adam optimizer, LR 0.01, momentum 0.937, weight decay 0.0005, pretrained backbone weights.

**Metrics**
Precision, Recall, mAP@0.5, mAP@0.5:0.95, F1, FLOPs, Parameters (model size), FPS, training time.

**Results**
- Maize-YOLO: **mAP@0.5 76.3%**, Recall **77.3%**, Precision 73.3%, F1 75.2%, **FLOPs 38.9G**, **FPS 67**, 33.4M parameters — a strong balance vs. RetinaNet, Faster R-CNN, YOLOv3–v7, YOLOR and YOLO-Lite variants (e.g., YOLOR-E6 had similar mAP but ~4× the FLOPs).
- Ablation: CSPResNeXt-50 alone: −61% FLOPs, +4% mAP vs YOLOv7 (but FPS −8). VoVGSCSP alone: −11% FLOPs, +4.5% mAP, +14 FPS. Combined (Maize-YOLO): −10% parameters, **−63% computation**, **+5.5% mAP**, **+10 FPS** vs YOLOv7.
- Per-class AP ranged from 40.6% (Yellow cutworm, hard due to larval/adult appearance shift) to 99.7% (Mole cricket).

---

## Paper 3 — RED-BIYO: Real-Time Red Palm Weevil Detection in Coconut Trees via Multimodal Sensor Data
*Siron Anita Susan T. et al. — International Journal of Computational Intelligence Systems, 2026*

**Problem Statement**
Red Palm Weevil (RPW) infestation causes severe, often irreversible damage to coconut trees. Larval damage is internal and invisible externally in early stages, so vision-only or sound-only detection methods have low sensitivity to early infestation, delaying intervention.

**Objectives**
- Detect RPW infestation early and reliably by fusing **visual (stem image)** and **acoustic (larval sound)** modalities.
- Provide an end-to-end IoT pipeline that alerts farmers in real time.

**Research Gaps Identified**
- Prior DL/ML/IoT frameworks for coconut disease/RPW detection rely on a single modality (image-only or sensor-only), each with blind spots (e.g., image methods miss internal larval activity; sensor-only methods lack spatial/visual context).
- Existing techniques largely detect late-stage, externally visible damage.

**Gaps Covered**
- Combines a **YOLOv9-based visual detector** (Damaged Tree/No Damaged Tree) with an **SCNN-BiGRU acoustic classifier** (RPW/No RPW), fused through a **Mamdani fuzzy inference system**, delivered via an IoT (Blynk) alert app — addressing both early (acoustic) and visible (image) infestation cues jointly.

**Methodology**
- **Image branch:** Savitzky-Golay (SavG) filter denoises stem images (checked via correlation, COR→1.0). YOLOv9 backbone uses ResNeSt split-attention bottlenecks + FPN/PAN neck for multi-scale detection of DT/NDT; loss combines localization, objectness, classification terms.
- **Audio branch:** Signal windowed with Hamming window, 512-point STFT → magnitude spectrogram → min-max normalized. Spectrogram converted to spike trains (rate coding, T=10) fed to a **Spiking CNN** (LIF neurons, surrogate-gradient training) for spatial-frequency features, followed by a **Bidirectional GRU** for temporal (forward/backward) modeling, ending in softmax (RPW/NRPW).
- **Fusion:** Gaussian-membership fuzzification of P(DT) and P(RPW) into Low/Medium/High; Mamdani IF-THEN rules (min for AND, max aggregation, centroid defuzzification) yield a Healthy/Not-Healthy decision, above a 0.6 alert threshold, pushed to a Blynk mobile app.

**Setup**
- Dataset: privately collected, Tamil Nadu, India (June–Dec 2024). Stem images: 3,000 (1,500 DT / 1,500 NDT, 1024×768, expert-annotated). Acoustic: 2,400 clips (1,200 RPW/1,200 NRPW, 16 kHz, 10–30 s, INMP441 digital microphone).
- 5-fold cross-validation (80/10/10 train/val/test per fold), fixed seed=42.
- YOLOv9: 640×640 input, auto-anchor (k-means), 100 epochs, batch 16, SGD+momentum 0.937, LR 0.01 (cosine annealing).
- SCNN-BiGRU: 128×128 spectrogram input, 3D conv filters 32/64/128 (3×3×3), spike steps T=10, dropout 0.5, batch 32, Adam LR 0.001, 100 epochs, cross-entropy loss.
- Hardware: Spyder/Anaconda on Windows 10, Intel i7, 16 GB RAM, NVIDIA RTX 3060.

**Metrics**
Accuracy (ACC), Specificity (SPE), Precision (PRE), Recall (REC), F1, Matthews Correlation Coefficient (MCC), AUC/ROC, model size, inference time.

**Results**
- Overall RED-BIYO: **ACC 99.54%**, MCC 97.93%, SPE ~98%, PRE 93.92%, REC 92.04%, F1 94.87%.
- Per-class: DT ACC 99.64%, NDT 98.73%, RPW 99.82%, NRPW 99.98% — AUCs 0.975–0.99.
- YOLOv9 (image branch) beat SSD/Faster R-CNN/Mask R-CNN/YOLOv6 by 7.21%/6.46%/4.54%/1.85% ACC respectively.
- SCNN-BiGRU (audio branch) beat LSTM/BiGRU/BiLSTM/CNN-BiGRU by 2.22%/4.95%/3.43%/1.41% ACC respectively.
- 5-fold average: ACC 97.92±1.63%, SPE 96.30±1.44%, PRE 95.46±0.94%, REC 93.89±1.24%, F1 94.26±0.93% — stable, low variance.
- Vs. whole-pipeline baselines (YOLOv5, ResNet50, Mask R-CNN, InceptionResNetV2, YOLOv3, MIN-SVM): RED-BIYO improved overall accuracy by 2.75%/9.38%/2.61%/22.16%/8.54%/4.39% respectively (p = 0.007, statistically significant).
- Smallest model size (68.9 MB) and lowest inference time (24.3 s/sample-batch metric) among compared networks.
- **Ablation:** image-only 98.12% ACC, audio-only 97.05% ACC, fused (both) 99.54% ACC — confirming multimodal fusion adds value over either single modality.

---

## Paper 4 — YOLO Deep Learning Algorithm for Object Detection in Agriculture: A Review
*Kamalesh Kanna S. et al. — Journal of Agricultural Engineering, 2024*

**Problem Statement**
Agriculture needs fast, accurate object detection for crops, weeds, pests/diseases, and land/resource monitoring, but the rapidly evolving YOLO family (v1–v9+ and variants) has not been consolidated into a single reference explaining architecture evolution, evaluation metrics, and agricultural applications.

**Objectives**
- Explain the core principles behind YOLO (regression-based, single-stage detection) and trace its version history (YOLOv1 → YOLOv9, plus YOLOX/YOLOR/PP-YOLO branches).
- Summarize YOLO's key accuracy metrics (mAP, IoU, precision/recall, NMS).
- Survey YOLO applications across agriculture (crop/weed/pest/disease detection, satellite/UAV remote sensing, tree/fruit detection) and compile representative performance figures.
- Identify limitations and future research directions.

**Research Gaps Identified**
- Base YOLO struggles with small-object detection (e.g., rice, sorghum, maize kernels/tassels).
- Newer YOLO versions (v8/v9+) are not yet widely adopted in agricultural practice.
- Multispectral-YOLO integration, robotics integration, and transfer learning to new crops/domains remain underexplored.
- No single review previously tied YOLO version evolution directly to agriculture-specific applications and metrics.

**Gaps Covered**
- This review itself compiles: (a) a structured YOLOv1–v9 architectural timeline (anchor boxes, PGI/GELAN, E-ELAN, CIoU/DFL losses, etc.), (b) a taxonomy of agricultural applications (crop monitoring, quality assessment, weed/disease/fruit/tree detection, satellite/UAV remote sensing) each mapped to a specific YOLO version and its reported metrics, and (c) a forward-looking table of research opportunities (real-time processing, multispectral imaging, robotics integration, transfer learning).

**Methodology**
This is a **literature review/synthesis** (not an experimental study): it collates and tabulates results from dozens of published agricultural-YOLO papers, organized by (i) YOLO version history and internal design changes, (ii) accuracy-metric definitions (mAP, confusion matrix, IoU, precision/recall, NMS), and (iii) application domain (crop monitoring, quality assessment, disease/weed/fruit/tree detection, remote sensing).

**Setup**
No original experiments/dataset/hardware — it references datasets and results reported in surveyed papers (e.g., IP102, PlantVillage, self-built UAV/satellite datasets, various YOLO versions v3–v9, YOLOX, YOLOR, GELAN).

**Metrics**
Discusses standard object-detection metrics used across surveyed works: mAP (mAP@0.5, mAP@0.5:0.95), Precision, Recall, F1-score, IoU, FPS, confusion matrix (TP/FP/TN/FN).

**Results (synthesized from surveyed studies)**
- YOLOv2: up to 78.6% mAP (VOC2007) at 40 FPS.
- YOLOv3: 36.2% AP / 60.6% AP50 (COCO) at ~2× speed of predecessor.
- YOLOv4: 43.5% AP, 65.7% AP50 (COCO, >50 FPS on V100).
- YOLOv5x: 50.7% AP (COCO).
- YOLOv7: 56.8% AP, fastest/most accurate real-time detector reported at the time (5–160 FPS range).
- YOLOv8x: 53.9% AP at 640px, 280 FPS on A100/TensorRT.
- Agricultural examples: YOLOv5 corn-kernel counting mAP0.5 0.70; YOLOv4 citrus disease mAP 95.4%; YOLOv5 tomato leaf disease 93% accuracy; YOLOv5s apple flower detection mAP 77.5%; YOLOv8L weed detection mAP0.5 0.9795; YOLOv4 grain counting accuracy 97.65%; YOLOv7 apple detection (multi-head) precision/recall/F1 0.91/0.96/0.92; Tassel-YOLO (modified YOLOv7) maize tassel mAP@0.5 96.14%, counting accuracy 97.55%.
- Concludes YOLO is accurate and fast for agriculture but still weak on small/rice/sorghum/maize objects; recommends continued architecture, multispectral, robotics, and transfer-learning research (summarized in a "future direction" table).

---

## Paper 5 — Towards Detecting Red Palm Weevil Using Machine Learning and Fiber Optic Distributed Acoustic Sensing
*Wang, Mao, Ashry, Al-Fehaid, Al-Shawaf, Ng, Yu & Ooi — Sensors, 2021*

**Problem Statement**
Red Palm Weevil (RPW) is the most destructive palm-tree pest globally, but palm trees show visible distress only in advanced, often unsalvageable infestation stages. Existing acoustic-probe methods are invasive, costly per-tree, and impractical at farm scale (thousands of trees); open-air noise (wind, birds) further complicates larval-sound detection.

**Objectives**
- Demonstrate early RPW detection using a **single non-invasive optical fiber** wound around trees, interrogated by a **distributed acoustic sensor (DAS)** system, scalable to large farms.
- Use neural networks (ANN and CNN) to classify healthy vs. infested trees from DAS-recorded temporal/spectral signals under realistic noise (wind, bird sound).
- Determine the best data representation (time-domain vs. frequency-domain) and best network architecture, and evaluate the impact of different optical-fiber jacket materials on noise robustness.

**Research Gaps Identified**
- X-ray tomography and trained dogs are accurate but too slow for large-scale farms.
- Per-tree acoustic probes are invasive, expensive, and impractical at scale.
- Prior DAS work (by the same group) used simple signal-processing thresholds only in controlled (noise-free) lab settings; open-air noise (wind-induced fiber vibration, bird sound) was not addressed.

**Gaps Covered**
- Introduces machine-learning (ANN/CNN) classification on top of DAS signals to handle realistic noise conditions (wind + bird sound), rather than simple thresholding.
- Compares two fiber jacket types (900 µm "JKT1" vs. 5 mm "JKT2") for wind-noise mitigation.
- Systematically evaluates temporal vs. spectral input representations for both ANN and CNN.

**Methodology**
- **DAS hardware:** φ-OTDR (phase-sensitive optical time-domain reflectometry) system — narrow-linewidth laser (100 Hz linewidth, 40 mW), AOM-pulsed (50 ns pulse, 5 kHz rep rate, 5 m spatial resolution), EDFA amplification, FBG noise filtering, photodetector, 200 MHz digitizer; ~2 km SMF with a 5 m coil around the tree trunk at ~1 km fiber distance.
- **Signal processing:** normalized differential method + FFT to locate the infested tree and extract larvae-sound frequency (dominant ~400 Hz); a [200–800 Hz] band-pass filter removes low-frequency wind/tree-swing noise and high-frequency electronic noise.
- **Labeling:** SNR (RMS at tree position vs. reference section) > 2 dB = "infested."
- **ANN:** fully-connected, 2 hidden layers (500, 50 nodes), ReLU + sigmoid output; input = 47,500-length temporal vector or 23,750-length spectral (FFT) vector per trial.
- **CNN:** 2D input (10 spatial points × 4750 temporal or × 2375 spectral), 2 conv+maxpool blocks (32 then 64 filters), FC 500→50, sigmoid output.

**Setup**
- Lab-simulated farm: infested tree (loudspeaker playing ~12-day-old larvae sound inside trunk, intensity 51–75 dB) and healthy tree; external fan (wind, ~3 m/s) and second loudspeaker (bird sound) as noise sources ~1 m away; background lab noise ~51 dB.
- 2,000 "infested" + 2,000 "healthy" examples per scenario (without wind; with wind); combined dataset = 8,000 examples.
- Split 60% train / 20% val / 20% test.
- Two fiber jackets compared: JKT1 (900 µm, Thorlabs SMF-28-J9) vs JKT2 (5 mm, YOFC-SCTX3Y-2B1-5.0-BL).

**Metrics**
Accuracy, Precision, Recall, False Alarm rate (from confusion matrix: TP/TN/FP/FN); training/validation accuracy & loss curves.

**Results**
- JKT2 (thicker jacket) was far more wind-noise-resistant than JKT1 (JKT1 shook within the 200–800 Hz band under wind, mimicking larvae sound; JKT2 rarely did) and gave a lower overall noise floor — so JKT2 was used for subsequent ML analysis. Neither jacket picked up bird sound (air attenuates it before reaching the fiber).
- **ANN:** temporal data gave poor generalization (83.6% accuracy, val ≪ train accuracy — overfitting/shift sensitivity); spectral data performed excellently — 99.3% (no wind), 99.6% (with wind), **99.9% (combined dataset)**, with precision/recall/false-alarm of 99.9%/99.9%/0.1% for the combined case.
- **CNN:** performed strongly on **both** representations due to spatial invariance — temporal data: 100.0% (no wind), 99.9% (with wind), 99.7% (combined); spectral data: 99.3%/98.3%/99.1% for the same three scenarios. Recommended CNN + temporal data for real-world deployment (avoids costly FFT step) — combined-data result: 99.7% accuracy, 99.5% precision, 99.9% recall, 0.5% false alarm.
- Feasibility analysis: DAS can be extended to ~10 km sensing range (~1 m spatial resolution), potentially monitoring ~1,000 trees per unit; estimated system cost ≈ US$37,000, ≈US$37/tree (with a 5 m fiber section per tree costing only ~US$2), reducible further by sharing a portable DAS unit across farms.
