# Model card: `models/dr_effnet_b0.pt`

## Summary

| | |
|:--|:--|
| Task | Grade diabetic retinopathy on the 5-level ICDR scale (0 No DR, 1 Mild, 2 Moderate, 3 Severe NPDR, 4 Proliferative DR) from one colour fundus photograph |
| Architecture | EfficientNet-B0 (timm `efficientnet_b0.ra_in1k`, ImageNet-pretrained), new 5-way head, dropout 0.3 |
| Input | Fundus photograph, cropped to the disc, resized to 384 x 384, local-average-colour subtracted, masked outside the disc |
| Output | Five grade probabilities; severity score = probability-weighted mean grade; grade = severity score cut at four validation-tuned thresholds; P(referable) = P(grade >= 2) |
| Size | 4.0 M parameters, 8 MB (float16 weights) |
| Training data | 1,160 EyePACS images (658 patients) |
| Licence | Code MIT. The weights were trained on Kaggle EyePACS competition data; check those terms before any non-research use. |

## Intended use

Education and research on automated DR grading: a portfolio demonstration of a leakage-free medical-imaging pipeline.
**Not** intended for screening, diagnosis or any clinical decision. It has not been validated prospectively, on
other cameras or populations, or against a regulatory standard.

## Data

A 2,111-image subset of the EyePACS images from the Kaggle Diabetic Retinopathy Detection competition (2015), about
512 px, one clinician grade per image. Patients are split into train / validation / test so that no patient (and so
no pair of eyes) crosses splits. See [`data/README.md`](data/README.md).

The subset has many more DR cases than real screening populations: grade 0 is 25% of the test set here, but about
73% of the full EyePACS data. Accuracy, precision and negative predictive value will differ at real-world prevalence;
sensitivity, specificity and AUC transfer better.

## Evaluation

Held-out test split: 540 images from 339 patients, never used for training, model selection or threshold tuning.
Test-time augmentation averages the four flips of each image. 95% confidence intervals come from a patient-level
bootstrap (2,000 resamples). Full numbers: [`reports/metrics.json`](reports/metrics.json); summary in the README.

## Limitations and risks

* **Small, single-source training set.** 1,160 images from one programme (EyePACS, US primary care). Performance on
  other cameras, ethnicities or image quality is unknown.
* **Label noise.** EyePACS grades come from one grader per image, and graders disagree substantially on adjacent
  grades (especially 0 vs 1 and 2 vs 3).
* **Mild DR is hard.** Grade 1 means microaneurysms only, a few pixels wide at 384 px; expect most confusion between
  grades 0, 1 and 2.
* **Treatment artefacts.** Many grade 4 images in EyePACS show laser photocoagulation scars from earlier treatment;
  the model can learn to recognise treatment rather than active proliferative disease.
* **Single image, single eye.** The model ignores the other eye, patient history and image-quality checks a
  screening programme would use; ungradable images are not detected.
* **Grad-CAM is a coarse, post-hoc explanation.** It shows where the severity score was most sensitive, at 12 x 12
  feature-map resolution; it is not a lesion segmentation.
