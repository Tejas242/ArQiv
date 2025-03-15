# Installation Guide

Follow these steps to install and configure ArQiv.

## Prerequisites

- **Python 3.7+**  
- Minimum 4GB RAM (8GB recommended)  
- Stable internet connection

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/tejas242/ArQiv.git
   cd ArQiv
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK Data**
   In a Python shell or script:
   ```python
   import nltk
   nltk.download('stopwords')
   ```

5. **Configure Kaggle API (Optional)**
   - Install Kaggle CLI: `pip install kaggle`
   - Set up your credentials following [Kaggle’s instructions](https://github.com/Kaggle/kaggle-api).

6. **Run ArQiv**
   - **CLI:** `python cli.py`
   - **Streamlit Web App:**  
     ```bash
     cd streamlit
     streamlit run streamlit_app.py
     ```

You're now ready to explore ArQiv!
