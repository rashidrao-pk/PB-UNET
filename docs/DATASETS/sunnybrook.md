## Dataset preprocessing and reproducibility statement:

- The Sunnybrook Cardiac Dataset was processed from the locally downloaded DICOM images and expert manual contours. Anonymized DICOM patient identifiers were matched to the original contour identifiers using the supplied scd_patientdata.csv file, with numeric components of patient identifiers normalized to account for leading zeros.
- For each patient, the cine short-axis series containing all frame indices referenced by the inner-contour annotations was selected. Missing or ambiguous series matches caused preprocessing to stop. For patient SCD0003901 (SC-N-05), an explicit configuration override selected unnamed_4, a 160-frame acquisition. This acquisition shared the short-axis orientation of the named 140-frame acquisition, which did not contain annotated frame 160.
- The segmentation target was the left ventricular cavity, defined by the expert inner contours. Outer contours were not used. Contour coordinates were interpreted as image-space (x, y) positions, truncated to integer pixel coordinates, and rasterized as filled polygons at the original image resolution. Masks were saved as binary PNG images with background value 0 and foreground value 255.
- DICOM pixel intensities were converted to floating point, and the stored rescale slope and intercept were applied. Each image was independently normalized to the range [0, 255] using its minimum and maximum intensities, rounded, and saved as an 8-bit grayscale PNG. Constant-intensity images were assigned zero values before photometric adjustment. Images with MONOCHROME1 photometric interpretation were inverted after normalization.
- This procedure produced 805 paired images and masks from 45 patients. Each pair received a unique patient-prefixed filename. A manifest recorded the generated image and mask paths, patient identifier, source DICOM path, and source contour path. Raw source files were retained unchanged.
- During training, images were resized to 256 × 256 pixels using bilinear interpolation, while masks were resized using nearest-neighbor interpolation. Images were represented as three identical grayscale channels and scaled to [0, 1]. Masks were converted to binary floating-point tensors. Data augmentation was disabled in the default configuration.
- The default experiment used an image-level split with random seed 42, yielding 565 training, 120 validation, and 120 test images. These subsets are not patient-disjoint. The actual split CSV files and resolved training configuration were saved with each run.
- These steps define the preprocessing used for this implementation; equivalence to the historical study’s preprocessing has not been established.

`To reproduce preparation and verification from the repository root`:

```bash
python scripts/prepare_sunnybrook.py --config configs/paper_legacy.yaml

python scripts/check_dataset.py --config configs/paper_legacy.yaml --require-prepared
```
