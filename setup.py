from setuptools import setup, find_packages

setup(
    name='afstabpfn',
    version='2.0.7',
    python_requires='>=3.9',
    packages=find_packages(),  # Automatically discovers all packages recursively
    install_requires=[
        "torch>=2.1,<3",
        "scikit-learn>=1.2.0,<1.7",
        "typing_extensions>=4.4.0",
        "scipy>=1.11.1,<2",
        "pandas>=1.4.0,<3",
        "einops>=0.2.0,<0.9",
        "huggingface-hub>=0.0.1,<1",
    ],
    extras_require={
        'dev': [
            "pre-commit",
            "ruff",
            "mypy",
            "pytest",
            "onnx",
            "psutil",
            "mkdocs",
            "mkdocs-material",
            "mkdocs-autorefs",
            "mkdocs-gen-files",
            "mkdocs-literate-nav",
            "mkdocs-glightbox",
            "mkdocstrings[python]",
            "markdown-exec[ansi]",
            "mike",
            "black",
        ]
    },
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
    ],
    license='LICENSE',
    author='Noah Hollmann, Samuel Müller, Lennart Purucker, Arjun Krishnakumar, Max Körfer, Shi Bin Hoo, Robin Tibor Schirrmeister, Frank Hutter, Eddie Bergman, Leo Grinsztajn',
    author_email='noah.hollmann@charite.de, muellesa@cs.uni-freiburg.de, fh@cs.uni-freiburg.de',
    description='TabPFN: Foundation model for tabular data',
    long_description_content_type='text/markdown',
    url='https://github.com/priorlabs/tabpfn',
)
