# LIA-FDS: Few-Shot Manufacturing & Food Defect Segmentation






## Overview

In manufacturing and food-processing lines, collecting and labeling thousands of defect images for every new product SKU is slow and expensive (**the factory cold-start problem**). In addition, changes in factory floor lighting (glare, shadows, or camera exposure shifts) often cause standard segmentation models to produce false alarms.

**LIA-FDS (Local Illumination-Adaptive Few-Shot Defect Segmentation)** is a practical few-shot segmentation pipeline built on top of a frozen **DINOv2 (`ViT-S/14`)** backbone. Using only **$K \in \{1, 5, 10\}$ defective support images**, the model adapts to unseen rigid industrial products and organic food items while staying robust against sudden illumination shifts.

The pipeline is benchmarked across four categories spanning two industrial domains:
* **Industrial Rigid Packaging (`MVTec-AD`):** `bottle`, `capsule`
* **Food & Agricultural Quality Control (`VisA`):** `cashew`, `fryum`



## Key Highlights

* **High Anomaly Localization with 5 Samples ($K=5$):** Achieves **97.40% Pixel AUROC** (`0.9833` at $K=10$) and **0.6223 mIoU** averaged across 3 random seeds (`42, 123, 3407`), outperforming 5-shot `U-Net` (`69.29% AUROC`, `0.5382 mIoU`) and `SegFormer` (`91.67% AUROC`, `0.5534 mIoU`).
* **2.37x Higher Few-Shot Defect IoU:** Reaches **0.2803 Defect IoU** and **0.5279 Pixel AP** at $K=5$, more than doubling the 5-shot `U-Net` baseline (`0.1180 Defect IoU`, `0.2640 AP`).
* **Robust to Factory Lighting Changes:** Under severe brightness, contrast, and gamma shifts on test images ($K=5$), `LIA_A3_Ours` maintains **96.66% Pixel AUROC** and **0.4819 Pixel AP**, whereas `U-Net` drops to `0.1620 AP` and `SegFormer` drops to `0.2592 AP`.
* **Real-Time Conveyor Speed:** Runs at **12.72 ms per frame (~78.6 FPS)** on an NVIDIA L4 GPU, making it suitable for real-time in-line quality inspection.

## Pipeline Architecture (`v5 Pro`)

1. **Frozen Foundation Backbone (`DINOv2 ViT-S/14`):** Extracts rich $16 \times 16$ patch token embeddings from $224 \times 224$ input images (`embed_dim = 384`) without overfitting on small datasets.
2. **Local Illumination-Invariant Adapter (`LIA`):** A lightweight residual spatial-channel adapter trained via episodic meta-learning (`1,500` episodes). During training, an illumination-consistency cosine loss aligns features between clean images and photometrically perturbed views.
3. **Multi-Prototype Support Memory:** Uses K-Means clustering ($M=5$) on normal background tokens from the support set alongside foreground defect prototypes to model diverse normal surface textures.
4. **Target-Domain Support Adaptation:** Performs `45` fast gradient adaptation steps on the $K$-shot support set (using `Focal + Dice` loss) before evaluating on unseen query images.

## Quantitative Results

### 1. Few-Shot Benchmark Across $K \in \{1, 5, 10\}$ (Averaged Across 3 Seeds: `42, 123, 3407`)

Full raw numbers are available in [results/ensemble\_summary\_across\_seeds.csv](https://www.google.com/search?q=results/ensemble_summary_across_seeds.csv).

| Shots ($K$) | Model | mIoU | Defect IoU | Pixel AUROC | Pixel AP | Latency (ms) | Throughput (FPS) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **LIA_A2_SegOnly** | **0.5833** | **0.2008** | **0.9503** | **0.4709** | 12.67 | 78.9 |
| **1** | **LIA_A3_Ours** | 0.5515 | 0.1402 | 0.9414 | 0.4327 | 12.67 | 78.9 |
| **1** | SegFormer | 0.5204 | 0.0823 | 0.8920 | 0.3016 | 9.57 | 104.5 |
| **1** | Frozen_DINOv2 | 0.5167 | 0.0726 | 0.9293 | 0.3401 | 12.42 | 80.5 |
| **1** | U-Net | 0.5041 | 0.0569 | 0.6400 | 0.1680 | **3.55** | **281.7** |
| **5** | **LIA_A3_Ours** | **0.6223** | **0.2803** | 0.9740 | **0.5279** | 12.72 | 78.6 |
| **5** | **LIA_A2_SegOnly** | 0.6213 | 0.2766 | **0.9765** | 0.5269 | 12.69 | 78.8 |
| **5** | Frozen_DINOv2 | 0.5712 | 0.1755 | 0.9613 | 0.4227 | 12.42 | 80.5 |
| **5** | SegFormer | 0.5534 | 0.1456 | 0.9167 | 0.3694 | 9.57 | 104.5 |
| **5** | U-Net | 0.5382 | 0.1180 | 0.6929 | 0.2640 | **3.54** | **282.5** |
| **10** | U-Net | **0.6124** | 0.2588 | 0.7198 | 0.3672 | **3.54** | **282.5** |
| **10** | **LIA_A2_SegOnly** | 0.6076 | **0.2643** | 0.9817 | **0.5215** | 12.68 | 78.9 |
| **10** | **LIA_A3_Ours** | 0.6074 | 0.2624 | **0.9833** | 0.5188 | 12.72 | 78.6 |
| **10** | SegFormer | 0.5869 | 0.2096 | 0.9412 | 0.4316 | 9.57 | 104.5 |
| **10** | Frozen_DINOv2 | 0.5593 | 0.1536 | 0.9675 | 0.4534 | 12.42 | 80.5 |

### 2. Factory Illumination Stress-Test ($K=5$, Unseen Lighting Shift on Query Images)

To evaluate robustness under changing factory floor lighting, query images were subjected to synthetic brightness, contrast, and gamma perturbations while support images remained clean. Full results are saved in [results/illumination\_stress\_test\_K5.csv](https://www.google.com/search?q=results/illumination_stress_test_K5.csv).

| Shots ($K$) | Model | Stress-Test mIoU | Stress-Test Defect IoU | Pixel AUROC | Pixel AP | Latency (ms) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **5** | **LIA_A3_Ours** | 0.5907 | 0.2249 | **0.9666** | **0.4819** | 12.59 |
| **5** | **LIA_A2_SegOnly** | **0.5937** | **0.2341** | 0.9638 | 0.4671 | 12.57 |
| **5** | Frozen_DINOv2 | 0.5407 | 0.1235 | 0.9498 | 0.3448 | 12.43 |
| **5** | SegFormer | 0.5306 | 0.1068 | 0.8845 | 0.2592 | 9.59 |
| **5** | U-Net | 0.5128 | 0.0685 | 0.6248 | 0.1620 | **3.54** |

### Practical Note on Patch Resolution vs. Pixel Metrics

* **Why Pixel AUROC is 97.4%–98.3% while hard Defect IoU is ~28%:** `DINOv2 (ViT-S/14)` processes $224 \times 224$ images using $14 \times 14$ patches, producing a $16 \times 16$ feature grid. When bilinearly upsampled $14\times$ back to $224 \times 224$, predictions on ultra-thin hairline scratches (`capsule` and `cashew`, where defects occupy `<2%` of image pixels) have slightly thicker boundaries than 1-pixel ground-truth masks.
* As a result, probability ranking across pixels (**Pixel AUROC = 0.9740–0.9833** and **Pixel AP = 0.5279**) is very strong for defect detection and localization, while hard-thresholded `Defect IoU` (`0.2803` at $K=5$) still outperforms 5-shot `U-Net` (`0.1180`) by **+137.5%** and `SegFormer` (`0.1456`) by **+92.5%**.

## Performance & Latency Visualizations

### mIoU Trend Across Few-Shot Settings ($K = 1, 5, 10$)


### Inference Latency Comparison ($K = 10$)


### Accuracy vs. Latency Trade-Off


## Repository Structure

```text
LIA-FDS-FewShot-Manufacturing-Defect-Segmentation/
├── LIA_FDS_Final_Research_Notebook_v5_Pro.ipynb   # Full training, adaptation & evaluation notebook
├── README.md                                      # Project documentation & benchmark summary
├── requirements.txt                               # Python dependencies
├── results/
│   ├── ensemble_summary_across_seeds.csv          # Multi-seed benchmark results (K=1, 5, 10)
│   └── illumination_stress_test_K5.csv            # Lighting shift robustness results (K=5)
└── assets/
    ├── sample_mvtec_bottle_defect.png             # Sample image & ground-truth defect mask
    ├── miou_trend_across_k.png                    # mIoU curve across K shots
    ├── inference_latency_k10.png                  # Model inference latency comparison
    └── latency_vs_miou_tradeoff.png               # mIoU vs latency scatter plot
```

## How to Run

1. **Clone the repository and install dependencies:**
   ```bash
   git clone https://github.com/usmannazir2342/LIA-FDS-FewShot-Manufacturing-Defect-Segmentation.git
   cd LIA-FDS-FewShot-Manufacturing-Defect-Segmentation
   pip install -r requirements.txt
   ```
2. **Run the notebook:**
   Open [LIA\_FDS\_Final\_Research\_Notebook\_v5\_Pro.ipynb](https://www.google.com/search?q=LIA_FDS_Final_Research_Notebook_v5_Pro.ipynb) in Google Colab (L4/T4 GPU recommended) or a local CUDA environment and run all cells sequentially.
