# Installation Guide

This guide details the step-by-step process to install and configure ArQiv.

## Prerequisites

- **Python 3.7+** must be installed.
- A stable internet connection is necessary to download the ArXiv dataset.
- (Optional) Kaggle CLI for dataset download (if not available, ArQiv falls back to curl).

## Setup Steps

### 1. Clone the Repository
```bash
git clone <repository_url>
cd Fun/ArQiv
```

### 2. Create a Virtual Environment
It is strongly recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
```

### 3. Install Dependencies
Install all required Python packages:
```bash
pip install datasets nltk rich numpy scikit-learn kaggle sentence-transformers
```

### 4. Download NLTK Resources
In a Python shell, execute:
```python
import nltk
nltk.download('stopwords')
```

### 5. Configure Kaggle API
- Install Kaggle CLI:
  ```bash
  pip install kaggle
  ```
- Follow the [Kaggle CLI setup instructions](https://github.com/Kaggle/kaggle-api) to configure your API credentials.
- The application will automatically download the dataset if not already present.

If any issues arise during installation, refer to the [Troubleshooting](troubleshooting.md) section.
