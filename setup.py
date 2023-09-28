from setuptools import find_packages, setup

__version__ = "0.0.1"

setup(
    name="smart-sense-assessment2",
    version=__version__,
    packages=find_packages(),
    install_requires=[
        "scikit-learn",
        "PyPDF2",
        "pdfplumber",
        "nltk",
        "gensim",
        "transformers",
        "pandas",
        "torchvision",
        "numpy",
        ],
)
