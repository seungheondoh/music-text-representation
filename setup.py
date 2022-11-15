from setuptools import setup

setup(
    name="mtr",
    packages=["mtr"],
    install_requires=[
        'librosa >= 0.8',
        'torchaudio_augmentations==0.2.1', # for augmentation
        'numpy',
        'pandas',
        'einops',
        'sklearn',
        'wandb',
        'jupyter',
        'matplotlib',
        'omegaconf',
        'astropy',
        'transformers',
    ]
)