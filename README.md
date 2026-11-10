# Plant Leaf Segmentation and disease percentage classification MVP

A small MVP for generating leaf-region pseudo-labels with **SAM 2.1** as teacher model and fine-tuning a **U-Net student model**. The student model is intended to provide a lighter-weight alternative for leaf segmentation at inference time.
The teacher model generates annotations for a benchmark dataset PlantDoc and this weakly labeled annotation is used to fine-tune the much lighter weight U-Net Model.

## Pipeline

1. **Teacher annotation** — Notebook 1 downloads a subset of PlantDoc images and uses SAM 2.1 Automatic Mask Generation (AMG) to generate candidate masks.
2. **Pseudo-mask filtering** — Candidates are filtered by area, predicted IoU and stability score, then combined into one binary leaf-region mask per image.
3. **Student training** — Notebook 2 loads the image/mask pairs, resizes them to 256 × 256, and fine-tunes a one-channel U-Net using binary cross-entropy with logits.
4. **Evaluation and inference** — The notebook reports test loss, Dice and IoU, visualizes predictions, and includes a CIELAB-based tissue classification example.

## Notebooks

| Notebook | Purpose |
|---|---|
| `SAM2_Hiera_Large_Teacher_Segmentation_Model.ipynb` | Download PlantDoc images, generate SAM 2.1 AMG pseudo-masks, review masks and save the annotation manifest. |
| `UNet_Student_Train.ipynb` | Load pseudo-labelled data, fine-tune U-Net, evaluate segmentation and run inference. |

Run Notebook 1 first. Notebook 2 expects the generated `images/`, `masks/` and `annotations/manifest.json` directories.

## Dataset

The teacher notebook uses a subset of [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset), an image dataset for plant disease recognition.

The notebook samples up to 150 images (subject to availability and file-size filtering). These are **image-level disease-class images**, not manually annotated segmentation data. The generated masks are SAM-derived pseudo-labels and should be visually inspected before training.

Dataset paper: Singh et al. (2020), *PlantDoc: A Dataset for Visual Plant Disease Detection*. https://doi.org/10.1145/3371158.3371196

## Runtime and dependencies

Designed for Google Colab with a CUDA-enabled GPU. The teacher notebook records a Tesla T4 runtime. It installs SAM 2 from the official repository and downloads a SAM 2.1 checkpoint.

The student notebook uses PyTorch, NumPy, Pillow and Matplotlib. It also downloads a pretrained Carvana U-Net checkpoint and imports the U-Net implementation from `models.py`; ensure that this file is available in the runtime.

Exact package versions and GPU availability can affect reproducibility. The notebooks should be run in order in a fresh Colab runtime.

## Important limitations

- SAM masks are automatic pseudo-labels, not ground truth. AMG proposals can include partial leaves, overlapping regions or background.
- The current filtering thresholds are heuristic and have not been validated as optimal.
- The combined binary mask represents the leaf region; it does not separate individual leaves.
- The dataset is small and sampled from PlantDoc. Results should be treated as an MVP demonstration, not a validated benchmark.
- The CIELAB tissue classes use fixed colour thresholds and are a separate heuristic stage; they are not learned by the U-Net.
- The current train/validation/test split is random at image level. It does not establish generalization to unseen plant species, acquisition conditions or datasets.

## Suggested execution

1. Open Notebook 1 in Google Colab and select a GPU runtime.
2. Run the cells to download images and generate masks.
3. Inspect the mask-review visualizations and check the saved manifest.
4. Open Notebook 2 and update `PROJECT_DIR` to the location of the generated data.
5. Ensure `models.py` and the pretrained U-Net checkpoint are accessible, then train and evaluate.

## License and attribution

Check the PlantDoc dataset license and the licenses/terms of the SAM 2 code, checkpoint, U-Net implementation and pretrained Carvana weights before redistributing data, weights or derived artifacts. This repository's code license should be specified separately by its maintainers.
