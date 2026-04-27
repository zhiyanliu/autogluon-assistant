# Condensed: More about CLIP

Summary: This tutorial demonstrates zero-shot image classification using CLIP via AutoGluon's `MultiModalPredictor` with `problem_type="zero_shot_image_classification"`. It covers initializing the predictor, using `predict_proba` with `{"image": [paths]}` and `{"text": [candidate_labels]}` dicts to classify images without training data, and downloading images via `autogluon.multimodal.utils.download`. Useful for tasks requiring label-free image classification against arbitrary text categories. Key warning: CLIP is vulnerable to typographic attacks where overlaid text can override visual predictions.

*This is a condensed version that preserves essential implementation details and context.*

# Zero-Shot Image Classification with CLIP (AutoGluon)

## Setup & Basic Usage

```python
!pip install autogluon.multimodal
```

```python
from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.utils import download

# Initialize zero-shot classifier
predictor = MultiModalPredictor(problem_type="zero_shot_image_classification")

# Classify image against text labels
dog_image = download("https://farm4.staticflickr.com/3445/3262471985_ed886bf61a_z.jpg")
prob = predictor.predict_proba(
    {"image": [dog_image]},
    {"text": ['This is a Husky', 'This is a Golden Retriever', 'This is a German Sheperd', 'This is a Samoyed.']}
)
```

**Key API pattern:** `predict_proba` takes a dict with `"image"` (list of paths) and a dict with `"text"` (list of candidate labels). No training data needed—just provide candidate categories.

CLIP works on uncommon classes too (e.g., segways), since it was pre-trained on 400M image-text pairs using contrastive learning to match images with their paired text.

```python
segway_image = download("https://live.staticflickr.com/7236/7114602897_9cf00b2820_b.jpg")
prob = predictor.predict_proba(
    {"image": [segway_image]},
    {"text": ['segway', 'bicycle', 'wheel', 'car']}
)
```

## Known Limitation: Typographic Attacks

⚠️ **CLIP is vulnerable to typographic attacks.** Adding text to an image can override visual classification:

```python
# Clean apple image → correctly predicts "Granny Smith" (~100% confidence)
apple = download("https://cdn.openai.com/multimodal-neurons/assets/apple/apple-blank.jpg")
prob = predictor.predict_proba({"image": [apple]}, {"text": ['Granny Smith', 'iPod', 'library', 'pizza', 'toaster', 'dough']})

# Apple with "iPod" text overlay → misclassifies as "iPod"
apple_ipod = download("https://cdn.openai.com/multimodal-neurons/assets/apple/apple-ipod.jpg")
prob = predictor.predict_proba({"image": [apple_ipod]}, {"text": ['Granny Smith', 'iPod', 'library', 'pizza', 'toaster', 'dough']})
```

For more details on limitations, see the [CLIP paper](https://arxiv.org/abs/2103.00020). For customization, refer to [Customize AutoMM](../advanced_topics/customization.ipynb).