# Plant Leaf Segmentation and Disease Percentage Classification MVP

A small MVP that uses **SAM 2.1 as a teacher** to generate leaf-region pseudo-labels, then fine-tunes a **U-Net student** to provide a lighter-weight segmentation model for inference.

The segmentation output can be used as the input to a separate **CIELAB-based tissue analysis** stage for estimating healthy, chlorotic and necrotic tissue percentages.

## Pipeline

1. **Input images** — A subset of PlantDoc images is used for the experiment.
2. **Teacher annotation (SAM 2.1)** — Automatic Mask Generation (AMG) produces candidate masks.
3. **Pseudo-mask filtering** — Candidates are filtered by area, predicted IoU and stability score. Selected masks are combined into one binary leaf-region mask per image.
4. **Student training (U-Net)** — The image/mask pairs are used to fine-tune a one-channel U-Net with binary cross-entropy with logits.
5. **Segmentation inference** — The trained student predicts a leaf-region probability map, which is thresholded to produce a binary mask.
6. **Downstream tissue analysis** — The leaf mask can be combined with CIELAB colour rules to estimate healthy, chlorotic and necrotic tissue percentages.

## Example outputs

### Input image

Example leaf image used to illustrate the segmentation task.

![Example input leaf](assets/beech_leaf_input.jpg)

### Teacher output — SAM 2.1 pseudo-label

SAM 2.1 generates candidate masks. After filtering and combining selected proposals, the teacher produces a binary leaf-region pseudo-label. The example below shows the leaf region overlaid on the original image.

![SAM 2.1 teacher annotation](assets/teacher_annotation_example.jpg)

### Student output — U-Net prediction

The student predicts a per-pixel probability map. Applying a threshold (0.5 in this example) converts the probability map into a binary segmentation mask.

![U-Net student prediction probability and thresholded mask](assets/student_prediction_example.png)

These examples are qualitative illustrations. They do not constitute a quantitative comparison against manually annotated ground truth.

## Notebooks

| Notebook | Purpose |
|---|---|
| `SAM2_Hiera_Large_Teacher_Segmentation_Model.ipynb` | Download a PlantDoc subset, generate SAM 2.1 AMG proposals, filter and combine masks, and save the annotation manifest. |
| `UNet_Student_Train.ipynb` | Load the generated image/mask pairs, fine-tune U-Net, evaluate segmentation and run inference. |

Run the teacher notebook first. The student notebook expects the generated image, mask and annotation directories.

## Dataset

The teacher notebook uses a subset of [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset), an image dataset for plant disease recognition.

The notebook samples up to 150 images, subject to availability and file-size filtering. PlantDoc provides image-level disease labels, not pixel-level segmentation ground truth. The masks generated here are therefore **SAM-derived pseudo-labels** and should be visually inspected before student training.

Dataset paper: Singh et al. (2020), *PlantDoc: A Dataset for Visual Plant Disease Detection*. https://doi.org/10.1145/3371158.3371196

## Runtime and dependencies

Designed for Google Colab with a CUDA-enabled GPU.

- **Teacher:** SAM 2.1, PyTorch and the dependencies specified by the official [SAM 2 repository](https://github.com/facebookresearch/sam2).
- **Student:** PyTorch, NumPy, Pillow and Matplotlib.
- **Student model code:** `models.py`.
- **Pretrained initialization:** the student notebook also uses a pretrained Carvana U-Net checkpoint; make the checkpoint accessible in the Colab runtime.

Run both notebooks in order in a fresh Colab runtime. Exact package versions and GPU availability can affect reproducibility.

## Important limitations

- SAM masks are automatic pseudo-labels, not ground truth. AMG proposals may include partial leaves, overlapping regions or background.
- The filtering thresholds are heuristic and have not been established as optimal.
- The combined binary mask represents a leaf region; it does not distinguish individual leaves.
- The dataset subset is small. Results should be treated as an MVP demonstration, not a validated benchmark.
- The CIELAB tissue classes use fixed colour thresholds and are a separate heuristic stage; they are not learned by U-Net.
- A random image-level train/validation/test split does not establish generalization to unseen plant species, acquisition conditions or datasets.

## Suggested execution

1. Open the teacher notebook in Google Colab and select a GPU runtime.
2. Generate the SAM 2.1 pseudo-masks.
3. Inspect the saved masks and annotation manifest.
4. Open the student notebook and set `PROJECT_DIR` to the generated data location.
5. Ensure `models.py` and the pretrained U-Net checkpoint are accessible.
6. Train the student, inspect probability maps and thresholded masks, and review the evaluation metrics.

## Repository files

```text
.
├── SAM2_Hiera_Large_Teacher_Segmentation_Model.ipynb
├── UNet_Student_Train.ipynb
├── models.py
├── README.md
└── assets/
    ├── beech_leaf_input.jpg
    ├── teacher_annotation_example.jpg
    └── student_prediction_example.png
```

## License and attribution

Check the PlantDoc dataset license and the licenses/terms of the SAM 2 code, checkpoint, U-Net implementation and pretrained Carvana weights before redistributing data, weights or derived artifacts. Specify a separate license for this repository's code.
