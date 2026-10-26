# Face Verification with FaceNet + Anti-Spoofing

A face verification pipeline that decides whether two images show the same person, plus a liveness check that flags printed-photo or screen attacks. It is built on a pretrained **FaceNet** model and evaluated on the **LFW (Labeled Faces in the Wild)** benchmark.

## Results

Evaluated on LFW match / mismatch test pairs:

| Metric | Value |
|---|---|
| Accuracy | **98.60%** |
| ROC AUC | **0.9957** |
| Best distance threshold | **1.03** (Euclidean, on FaceNet embeddings) |

Example outputs from the notebook:

| Pair | Distance | Result |
|---|---|---|
| Same person (Abdullah, image 1 vs 2) | 0.6820 | Same person |
| Different people (Abdullah vs Colin Powell) | 1.5563 | Different person |

## How it works

### 1. Face verification
1. **Detect and align:** `MTCNN` finds the face and crops it to 160×160 with a 20px margin.
2. **Embed:** `InceptionResnetV1` (pretrained on **VGGFace2**) turns the face into an embedding vector.
3. **Compare:** the Euclidean distance between the two embeddings is computed. Distance below the threshold means the same person.
4. **Confidence (heuristic):** `(1 - distance / 2) * 100`. This is a readable score, not a calibrated probability.

### 2. Threshold selection
The verification threshold is chosen by sweeping values from 0.5 to 1.5 in steps of 0.01 over all evaluated pairs and keeping the one with the highest accuracy (1.03). The notebook also plots the ROC curve.

### 3. Anti-spoofing (liveness)
Uses [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) by Minivision. Two lightweight models, `MiniFASNetV1SE` and `MiniFASNetV2`, each predict on a cropped patch of the face. Their outputs are summed and the highest class decides: **real** or **spoof**.

> **Note:** LFW images are photographs, so the anti-spoofing model correctly flags them as spoofs. That result is expected on this dataset. To test liveness properly, run the check on a live webcam feed and compare a real face with a person holding up a photo.

## Datasets

- [LFW cropped faces](https://www.kaggle.com/datasets/jonathanloscalzo/lfw-cropped-faces): the face images
- [LFW dataset](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset): the match and mismatch pair CSVs (`matchpairsDevTest.csv`, `mismatchpairsDevTest.csv`)

## Installation

```bash
pip install -r requirements.txt
```

The anti-spoofing part also needs the Minivision repo, which holds the model weights and helper code:

```bash
git clone https://github.com/minivision-ai/Silent-Face-Anti-Spoofing.git
cd Silent-Face-Anti-Spoofing
```

A GPU is used automatically when available (`cuda`), otherwise the code runs on CPU.

## Usage

Run the notebook `face-verifcation-facenet-antispoofing.ipynb` top to bottom (it was developed on Kaggle).

**Verify two images**
```python
verified, distance, confidence = verify_faces("img1.jpg", "img2.jpg", threshold=1.03)
```

**Check liveness** (run from inside the `Silent-Face-Anti-Spoofing` folder)
```python
check_spoofing("face.jpg")   # True = real face, False = spoof
```

## Project structure

```
├── face-verifcation-facenet-antispoofing.ipynb   # Full pipeline: verification, evaluation, anti-spoofing
├── requirements.txt
└── README.md
```

## Limitations

- The threshold (1.03) was tuned on the same LFW test pairs used for reporting accuracy, so the 98.60% figure is slightly optimistic. Tuning on a separate validation split would give a cleaner estimate.
- LFW is mostly frontal, well-lit celebrity photos. Performance on webcam or low-light images may differ.
- Anti-spoofing was only demonstrated on a still image. It has not been benchmarked on live attack data.

## Roadmap

- [ ] Real-time webcam demo combining verification and liveness
- [ ] Separate validation split for threshold tuning
- [ ] Wrap the pipeline in a REST API
- [ ] Add a face-attendance application on top

## Author

**Abdelrahman**, ML / Computer Vision engineer based in Cairo, Egypt.
