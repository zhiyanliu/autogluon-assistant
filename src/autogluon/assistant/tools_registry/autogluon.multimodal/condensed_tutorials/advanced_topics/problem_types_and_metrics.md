# Condensed: Convert set to sorted list for complete display

Summary: This tutorial covers AutoGluon MultiModal's `PROBLEM_TYPES_REG` registry for programmatically querying supported problem types and their properties. It demonstrates how to use `PROBLEM_TYPES_REG.get()` with constants like `BINARY`, `MULTICLASS`, `REGRESSION`, `OBJECT_DETECTION`, `SEMANTIC_SEGMENTATION`, `TEXT_SIMILARITY`, `IMAGE_SIMILARITY`, `IMAGE_TEXT_SIMILARITY`, `NER`, `FEATURE_EXTRACTION`, and `FEW_SHOT_CLASSIFICATION` to inspect supported modalities, evaluation metrics, default metrics, zero-shot capability, and training support. Useful for dynamically selecting problem types, validating configurations, and building workflows that adapt based on available capabilities in AutoGluon's multimodal framework.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon MultiModal Problem Types Reference

## Setup

```python
!pip install autogluon.multimodal
```

```python
import warnings
warnings.filterwarnings('ignore')

from autogluon.multimodal.constants import *
from autogluon.multimodal.problem_types import PROBLEM_TYPES_REG
```

## Querying Problem Type Properties

Use `PROBLEM_TYPES_REG.get()` to inspect any problem type's supported modalities, metrics, and capabilities:

```python
def print_problem_type_info(name: str, props):
    print(f"\n=== {name} ===")
    print("\nSupported Input Modalities:")
    for modality in sorted(list(props.supported_modality_type)):
        print(f"- {modality}")
    if hasattr(props, 'supported_evaluation_metrics') and props.supported_evaluation_metrics:
        print("\nEvaluation Metrics:")
        for metric in sorted(list(props.supported_evaluation_metrics)):
            if metric == props.fallback_evaluation_metric:
                print(f"- {metric} (default)")
            else:
                print(f"- {metric}")
    if hasattr(props, 'support_zero_shot'):
        print(f"\nZero-shot: {'Supported' if props.support_zero_shot else 'No'}")
        print(f"Training: {'Supported' if props.support_fit else 'No'}")
```

## Supported Problem Types

| Problem Type | Constant | Key Notes |
|---|---|---|
| **Binary Classification** | `BINARY` | 2 classes |
| **Multiclass Classification** | `MULTICLASS` | 3+ classes |
| **Regression** | `REGRESSION` | Numerical prediction |
| **Object Detection** | `OBJECT_DETECTION` | Bounding box localization |
| **Semantic Segmentation** | `SEMANTIC_SEGMENTATION` | Pixel-level classification |
| **Text Similarity** | `TEXT_SIMILARITY` | Text-to-text matching |
| **Image Similarity** | `IMAGE_SIMILARITY` | Image-to-image matching |
| **Image-Text Similarity** | `IMAGE_TEXT_SIMILARITY` | Cross-modal matching |
| **NER** | `NER` | Entity classification in text |
| **Feature Extraction** | `FEATURE_EXTRACTION` | Raw data → feature vectors |
| **Few-shot Classification** | `FEW_SHOT_CLASSIFICATION` | Small examples per class |

### Querying Examples

```python
# Classification
binary_props = PROBLEM_TYPES_REG.get(BINARY)
multiclass_props = PROBLEM_TYPES_REG.get(MULTICLASS)

# Similarity matching (all three support zero-shot)
for type_key in [TEXT_SIMILARITY, IMAGE_SIMILARITY, IMAGE_TEXT_SIMILARITY]:
    props = PROBLEM_TYPES_REG.get(type_key)
    # Access: props.supported_modality_type, props.support_zero_shot
```

**Key properties available:** `supported_modality_type`, `supported_evaluation_metrics`, `fallback_evaluation_metric`, `support_zero_shot`, `support_fit`.

## Further Resources

- **Similarity matching:** [Matching Tutorials](../semantic_matching/index.md)
- **More examples:** [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm)
- **Customization:** [Customize AutoMM](../advanced_topics/customization.ipynb)