# Data

The image datasets are **not** redistributed in this repository. Download them from their original sources:

| Dataset | Source | Used for |
|:--|:--|:--|
| Brain Tumor MRI Dataset, **version 2** (7,200 images; 1,400 training and 400 test images per class) | Kaggle: `masoudnickparvar/brain-tumor-mri-dataset` — https://doi.org/10.34740/kaggle/dsv/2645886 | training and evaluation |
| Brain tumor dataset of Cheng *et al.* (`.mat` files with patient IDs) | figshare — https://doi.org/10.6084/m9.figshare.1512427 | recovering patient identifiers only (not used for training) |

Expected layout:

```
brain-tumor-mri-dataset/
├── Training/
│   ├── glioma/  ├── meningioma/  ├── notumor/  └── pituitary/
└── Testing/
    ├── glioma/  ├── meningioma/  ├── notumor/  └── pituitary/
```

`scripts/s00_audit.py` prints the release it detects; it should report `v2 (7,200 images)`.
The Kaggle notebook in `notebooks/` downloads the figshare files automatically when Internet access is on.
