# Data

The images are not stored in this repository. They come from the **EyePACS** retinal images released for the
[Kaggle Diabetic Retinopathy Detection](https://www.kaggle.com/c/diabetic-retinopathy-detection) competition (2015),
whose terms do not allow redistribution. Each image is graded 0-4 on the International Clinical Diabetic Retinopathy
scale by a clinician.

This project uses a 2,111-image subset of the competition's training set (1,254 patients), resized to about
512 x 512 px, with the grades spread more evenly than in the full data:

| Grade | 0 No DR | 1 Mild | 2 Moderate | 3 Severe | 4 Proliferative | Images | Patients |
|:--|--:|--:|--:|--:|--:|--:|--:|
| train | 301 | 286 | 331 | 143 | 99 | 1,160 | 658 |
| val | 103 | 110 | 99 | 66 | 33 | 411 | 257 |
| test | 133 | 141 | 129 | 81 | 56 | 540 | 339 |

## `splits.csv`

One row per image: `image` (Kaggle file name without extension, `<patient>_<eye>`), `patient`, `eye`, `grade`,
`split` (train / val / test), and `original_split` (the folder the image came from).

The subset arrived pre-split into `training/`, `validation/` and `test/` folders, but split by image, so 49 patients
had one eye in training and the other in validation or test. `scripts/make_splits.py` moves those 49 images so that
**every patient is in exactly one split** (training wins over validation, validation over test).

## Layout expected by the scripts

`--source` points at the folder that holds the original split folders, each with one sub-folder per grade:

```
<source>/
├── training/   0/ 1/ 2/ 3/ 4/   *.jpeg
├── validation/ 0/ 1/ 2/ 3/ 4/   *.jpeg
└── test/       0/ 1/ 2/ 3/ 4/   *.jpeg
```

If you start from the full Kaggle download instead (`train/*.jpeg` + `trainLabels.csv`), copy the images listed in
`splits.csv` into that layout (`original_split` train / val / test → `training/`, `validation/`, `test/`; `grade` →
sub-folder) and resize them to about 512 px on the long side.

`scripts/preprocess_images.py` then writes the preprocessed copies used for training to `data/processed/<size>/`.
