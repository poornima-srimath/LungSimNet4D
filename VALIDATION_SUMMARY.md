THORIX Validation Summary

Anatomical Validation
PASS

Lesion Placement Validation
PASS

Voxelization Validation
PASS

GATE Geometry Validation
PASS

Material Mapping Validation
PASS

Photon Transport Validation
PASS

Detector Acquisition Validation
PASS

ROOT Output Validation
PASS

Segmentation Training Validation
PASS

Classification Training Validation
PASS

Classification Metrics

Accuracy: 0.80
Precision: 1.00
Recall: 0.60
F1 Score: 0.75

Dataset Size

1044 Anatomical Variants Generated

10 Voxelized Phantoms Used For Validation

Repository Validation Scope

The validation artifacts included in this repository demonstrate
end-to-end framework functionality using a representative subset
of generated phantoms.

Benchmark performance metrics reported in the manuscript were
obtained through separate experimental evaluations and are
reported independently from the framework validation subset.


---- LUNA 16 ----

LUNA16 External Validation Summary

Dataset Validation
PASS

Patch Extraction Validation
PASS

Train / Validation / Test Split Validation
PASS

MONAI ResNet-3D Training Validation
PASS

External Clinical Dataset Validation
PASS

Classification Metrics

Accuracy: 0.55
Precision: 0.55
Recall: 1.00
F1 Score: 0.71
ROC-AUC: 0.91

Dataset Size

366 Clinical CT Studies Used

756 Annotated Nodules

756 Positive Patches

756 Negative Patches

1512 Total Patches

Repository Validation Scope

The validation artifacts included in this repository demonstrate
external evaluation of the LungSimNet4D classification workflow
using a subset of the publicly available LUNA16 dataset.

The objective of this validation was to verify generalization
capability beyond synthetic THORIX-generated phantoms and
demonstrate compatibility with real-world clinical CT data.

The results reported here represent a limited-scale benchmark
executed under CPU-based training conditions. Additional
optimization, calibration, and large-scale evaluation remain
part of future work.