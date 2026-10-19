# CS 5330 Mini-Project 3: Curated VisA Dataset

This repository holds the image data for Mini-Project 3 in CS 5330 (Pattern Recognition & Computer Vision, Fall 2026, Northeastern University Vancouver). It is a small, curated copy of the **Visual Anomaly (VisA)** dataset, prepared so that you can load it in Google Colab in seconds instead of downloading the full 1.8 GB release.

Each image shows a food product that is either normal or defective. Defective images come with a pixel-level mask marking exactly where the defect is, so you can measure how well your system highlights the damaged region.

## Subsets

| Subset | Role | What you'll see |
| --- | --- | --- |
| `cashew` | Core (required) | One cashew per image, roughly aligned |
| `macaroni1` | Core (required) | Several pieces of macaroni per image |
| `chewinggum` | Extension | One piece of gum per image |
| `fryum` | Extension | One fryum (a puffed snack) per image |
| `pipe_fryum` | Extension | One pipe-shaped fryum per image |
| `macaroni2` | Extension | Several pieces of macaroni on a textured background; the hardest subset |

Everyone works with the two core subsets. The extension subsets are optional.

## Splits

Every subset has the same three splits:

| Split | Normal | Anomalous | What it's for |
| --- | --- | --- | --- |
| `reference` | 40 | 0 | A pool of defect-free images. Your few-shot method builds its memory bank from *k* of these. |
| `dev` | 20 | 20 | Tuning: choose prompts, thresholds, and other settings here. |
| `test` | 50 | 50 | Final results only. |

**Tune on `dev`, report on `test`.** Don't adjust prompts or thresholds after looking at test results. If you do, your test numbers no longer tell you how the system would perform on new images, and the live demo will expose that quickly.

No image appears in more than one split.

## Folder layout

```
data/
  cashew/
    reference/   cashew_reference_normal_000.jpg ...
    dev/         cashew_dev_anomaly_000.jpg, cashew_dev_anomaly_000_mask.png, cashew_dev_normal_000.jpg ...
    test/        ...
  macaroni1/
  ...
  manifest.csv
```

- Images are JPEG, resized so the longer side is 512 px.
- Masks are greyscale PNG files with the same size as their image: 0 = normal, 255 = defect. Only anomalous images have masks.

## The manifest

`data/manifest.csv` lists every image, one row each:

| Column | Meaning |
| --- | --- |
| `subset` | e.g. `cashew` |
| `split` | `reference`, `dev`, or `test` |
| `label` | `normal` or `anomaly` |
| `image_path` | Path to the image, relative to the repository root |
| `mask_path` | Path to the mask (empty for normal images) |
| `width`, `height` | Image size in pixels |
| `defect_area_px` | Number of defect pixels in the mask (empty for normal images) |
| `original_filename` | The image's path in the original VisA release |

## Loading the data in Colab

```python
!git clone --depth 1 https://github.com/LinoNU/cs5330-mp3-visa.git

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path("cs5330-mp3-visa")
manifest = pd.read_csv(REPO / "data" / "manifest.csv", keep_default_na=False)

dev_cashew = manifest.query("subset == 'cashew' and split == 'dev'")
row = dev_cashew[dev_cashew["label"] == "anomaly"].iloc[0]

image = Image.open(REPO / row["image_path"]).convert("RGB")
mask = np.asarray(Image.open(REPO / row["mask_path"])) == 255  # True where the defect is
print(image.size, mask.shape, mask.sum())
```

Note that `image.size` is (width, height) while `mask.shape` is (height, width).

## Attribution and licence

This data is a modified version of the VisA dataset, released by Amazon under the [Creative Commons Attribution 4.0 International licence (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). The original release is available at <https://github.com/amazon-science/spot-diff>.

Changes made for this course: six subsets were selected, each subset was resampled into new reference / dev / test splits, images and masks were resized to 512 px on the longer side, masks were converted to binary, and images were re-encoded as JPEG with all metadata removed. See `LICENSE-DATA` for details.

If you use this data in your report, cite the original paper:

```bibtex
@inproceedings{zou2022spot,
  title     = {SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation},
  author    = {Zou, Yang and Jeong, Jongheon and Pemula, Latha and Zhang, Dongqing and Dabeer, Onkar},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2022},
  note      = {arXiv:2207.14315}
}
```

## Regenerating the data (instructors)

You don't need this section to do the mini-project.

The curated data is produced by `scripts/prepare_visa.py`, which is deterministic: the same arguments always produce the same files.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

mkdir -p raw
aws s3 cp --no-sign-request s3://amazon-visual-anomaly/VisA_20220922.tar raw/
tar -xf raw/VisA_20220922.tar -C raw/

# --raw-dir must point to the folder that contains cashew/, macaroni1/, ...
python scripts/prepare_visa.py --raw-dir raw/<extracted-folder> --out-dir data --overwrite
```

The script checks that split counts are exact, that no image is reused across splits, that every mask is binary, matches its image size, and still contains a defect after resizing, and that no metadata remains. It also warns about defects smaller than 50 px after resizing (adjust with `--min-defect-px`). Run `python scripts/prepare_visa.py --help` for all options.
