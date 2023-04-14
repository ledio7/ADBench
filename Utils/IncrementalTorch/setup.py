import io
import os
import subprocess
import sys

import setuptools

try:
    from numpy import get_include
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
    from numpy import get_include

# Package meta-data.
NAME = "IncrementalTorch"
DESCRIPTION = "Online Deep Learning for river"
LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"
URL = "https://github.com/kulbachcedric/IncrementalTorch"
EMAIL = "cedric.kulbach@googlemail.com"
AUTHOR = "Cedric Kulbach"
REQUIRES_PYTHON = ">=3.6.0"

# Package requirements.
base_packages = [
    "scikit-learn==1.0.2",
    "scikit-surprise==1.1.1",
    "torch==1.10.2",
    "vaex==4.8.0",
    "pandas~=1.3.2",
    "numpy==1.22.2",
    "river~=0.10.1",
    "tqdm~=4.61.2",
    "pytest==7.0.1",
]

dev_packages = base_packages + [
    "graphviz>=0.10.1",
    "matplotlib>=3.0.2",
    "mypy>=0.761",
    "pre-commit>=2.9.2",
    "pytest>=4.5.0",
    "pytest-cov>=2.6.1",
    "scikit-learn>=0.22.1",
    "sqlalchemy>=1.4",
]

docs_packages = [
    "flask==2.0.2",
    "ipykernel==6.9.0",
    "mike==0.5.3",
    "mkdocs==1.2.3",
    "mkdocs-awesome-pages-plugin==2.7.0",
    "mkdocs-material==8.1.11",
    "mkdocstrings==0.18.0",
    "mkdocs-jupyter==0.20.0",
    "nbconvert==6.4.2",
    "numpydoc==1.2",
    "spacy==3.2.2",
]

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = "\n" + f.read()

# Load the package's __version__.py module as a dictionary.
about = {}
with open(os.path.join(here, NAME, "__version__.py")) as f:
    exec(f.read(), about)

# Where the magic happens:
setuptools.setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=setuptools.find_packages(exclude=("tests",)),
    install_requires=base_packages,
    extras_require={
        "dev": dev_packages,
        "docs": docs_packages,
        "all": dev_packages,# + docs_packages,
        ":python_version == '3.6'": ["dataclasses"],
    },
    include_package_data=True,
    license="BSD-3",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    ext_modules=[]
)
