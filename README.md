# Fashion-MNIST CNN

An end-to-end PyTorch notebook that develops three convolutional neural networks for Fashion-MNIST, moving from a compact baseline to a deeper architecture and then to a regularised, tuned model. The notebook combines implementation with an extended explanation of the design and experimental results.

## Covered techniques

- Fashion-MNIST download, visualisation and normalisation
- Reusable training, validation and test loops
- Learning-rate discovery with fastai
- Early stopping for the baseline models
- Basic, deeper and advanced CNN architectures
- Dropout, batch normalisation and global average pooling
- Training-only data augmentation
- SGD with momentum and weight decay
- Cosine-annealing learning-rate scheduling
- Confusion matrices and per-class error analysis
- Grad-CAM visualisation of learned attention
- Accuracy and training-time comparison across models

## Requirements

- Python 3
- Jupyter Notebook or JupyterLab
- PyTorch and Torchvision
- fastai
- `grad-cam`
- scikit-learn
- NumPy, Matplotlib and Seaborn

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install jupyter torch torchvision fastai grad-cam scikit-learn numpy matplotlib seaborn
jupyter notebook Optimizing_Convolutional_Neural_Networks_An_Iterative_Approach_to_Hyperparameter_Tuning_for_Image_Classification.ipynb
```

Fashion-MNIST is downloaded automatically by Torchvision. CUDA is used when available, but the notebook also supports CPU execution. Full training includes several models of up to 100 epochs, so CPU-only runs can take a long time.

## Model progression

1. **Basic CNN** — establishes the training pipeline, evaluation and visualisation tools.
2. **Deeper CNN** — adds convolutional capacity to improve feature extraction.
3. **Advanced CNN** — combines deeper blocks with batch normalisation, dropout, augmentation, SGD and cosine annealing.

## Status

This is a tutorial-style experimental notebook rather than a deployable inference service. Outputs are retained in the notebook so the analysis can be read without rerunning the complete training process.
