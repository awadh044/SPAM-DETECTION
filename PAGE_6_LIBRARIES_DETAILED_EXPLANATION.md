# 📚 PAGE 6: Libraries and Dependencies - DETAILED EXPLANATION

## Overview
This page explains all the Python libraries used in the spam detection project and why each one is essential.

---

## 🎯 What are Libraries?

**Libraries** (also called packages/modules) are pre-written code collections that provide ready-made functions and tools. Instead of writing everything from scratch, we use libraries to save time and leverage proven, tested code.

---

## 📦 All Libraries Used in This Project

### 1. **NumPy** 🔢

#### What is it?
A powerful library for **numerical computing** and working with arrays.

#### Why We Use It
- Performs fast mathematical operations
- Handles large arrays and matrices efficiently
- Foundation for other data science libraries

#### Code Example
```python
import numpy as np

# Create arrays
array1 = np.array([1, 2, 3, 4, 5])
array2 = np.array([10, 20, 30, 40, 50])

# Quick calculations
result = array1 + array2  # [11, 22, 33, 44, 55]
print(result)

# Statistics
mean_value = np.mean(array1)  # 3.0
max_value = np.max(array2)    # 50
```

#### Where Used in Project
Behind the scenes in Pandas and Scikit-learn operations.

#### Installation
```bash
pip install numpy
```

---

### 2. **Pandas** 📊

#### What is it?
A library for **data manipulation and analysis** - works with spreadsheet-like data (DataFrames).

#### Why We Use It
- Load data from CSV files
- Organize data in tables
- Clean and transform data easily
- Analyze data with built-in functions

#### Code Example
```python
import pandas as pd

# Load CSV file
df = pd.read_csv('spam.csv')

# View first 5 rows
print(df.head())

# Get column information
print(df.info())

# Statistics
print(df.describe())

# Count spam vs ham
print(df['label'].value_counts())

# Filter data
spam_messages = df[df['label'] == 'spam']
print(f"Found {len(spam_messages)} spam messages")
```

#### Dataset Structure
```
   label               message
0    ham  Go until jurong point
1    ham  Ok lar. Joking wif u
2   spam  Free entry in 2 a wkly
3    ham  U dun say so early
4    ham  Nah I don't think he
```

#### Where Used in Project
**Primary use** in `data_preparation.py`:
- Load the SMS spam dataset
- Preprocess and clean messages
- Split into training and testing sets

#### Installation
```bash
pip install pandas
```

---

### 3. **Scikit-learn** 🤖

#### What is it?
A comprehensive **machine learning library** with algorithms, feature extraction, and evaluation tools.

#### Why We Use It
- Implements Naive Bayes algorithm
- Provides TF-IDF vectorizer
- Offers metrics for evaluation
- Easy train-test splitting

#### Core Components Used

##### a) **Multinomial Naive Bayes**
```python
from sklearn.naive_bayes import MultinomialNB

# Create model
model = MultinomialNB()

# Train on data
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Get probabilities
probabilities = model.predict_proba(X_test)
```

**What it does**: 
- Learns probability of words in spam vs. ham messages
- Makes predictions based on learned probabilities

##### b) **TF-IDF Vectorizer**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Create vectorizer
vectorizer = TfidfVectorizer(
    max_features=5000,      # Keep top 5000 words
    min_df=2,               # Word must appear ≥2 times
    max_df=0.8,             # Word can appear ≤80% of docs
    ngram_range=(1, 2)      # Single words + word pairs
)

# Fit and transform training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform test data
X_test_tfidf = vectorizer.transform(X_test)
```

**What it does**:
- Converts text words to numerical values (features)
- Creates matrix of word importance scores
- Output: 4457 messages × 5000 features

##### c) **Train-Test Split**
```python
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,           # 20% for testing
    random_state=42,         # Reproducible results
    stratify=y               # Maintain class ratio
)
```

**What it does**:
- Divides data: 80% for training, 20% for testing
- Ensures fair evaluation on unseen data

##### d) **Evaluation Metrics**
```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Get detailed report
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
```

#### Where Used in Project
**Multiple places**:
- `feature_extraction.py` → TF-IDF vectorization
- `naive_bayes_classifier.py` → Model training & evaluation
- `main.py` → Orchestrating pipeline

#### Installation
```bash
pip install scikit-learn
```

---

### 4. **NLTK** (Natural Language Toolkit) 🔤

#### What is it?
A library for **natural language processing** - tools for text analysis and manipulation.

#### Why We Use It
- Tokenize text (break into words)
- Remove stopwords (common words like "the", "is")
- Stem words (reduce to root form: "running" → "run")

#### Code Example
```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required data (first time only)
nltk.download('punkt')
nltk.download('stopwords')

# Sample text
text = "You've won a FREE iPhone! Click here NOW!!!"

# 1. TOKENIZATION - Break into words
tokens = word_tokenize(text)
print(tokens)
# Output: ["You", "'ve", "won", "a", "FREE", "iPhone", "!", "Click", "here", "NOW", "!"]

# 2. STOPWORDS - Remove common words
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
print(filtered_tokens)
# Output: ["You", "'ve", "won", "FREE", "iPhone", "!", "Click", "here", "NOW", "!"]

# 3. STEMMING - Reduce to root form
stemmer = PorterStemmer()
stemmed = [stemmer.stem(word) for word in filtered_tokens]
print(stemmed)
# Output: ["you", "'ve", "won", "free", "iphon", "!", "click", "here", "now", "!"]
```

#### Processing Pipeline in Project
```
Raw Text
    ↓
Word Tokenization (split into words)
    ↓
Lowercase (convert to same case)
    ↓
Remove Non-alphabetic (remove !, ?, numbers)
    ↓
Remove Stopwords (remove: the, is, a, an)
    ↓
Stemming (running→run, walked→walk)
    ↓
Clean Text Ready for ML Model
```

#### Common Stopwords Removed
```
the, is, at, which, on, and, or, if, in, a, an, as, be, 
by, for, from, has, he, I, it, its, of, that, to, was, will, with
```

#### Common Stopwords NOT Removed (Important for Spam Detection)
- "free" → indicates spam (kept)
- "click" → indicates spam (kept)
- "win" → indicates spam (kept)

#### Where Used in Project
**Main use** in `data_preparation.py`:
```python
class DataPreparation:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        # All NLTK processing happens here
        pass
```

#### Installation
```bash
pip install nltk
```

---

### 5. **Matplotlib** 📈

#### What is it?
A library for **creating visualizations** - graphs, charts, and plots.

#### Why We Use It
- Create publication-quality visualizations
- Plot confusion matrix
- Display model performance
- Save figures as PNG/JPG

#### Code Example
```python
import matplotlib.pyplot as plt

# Simple line plot
months = ['Jan', 'Feb', 'Mar', 'Apr']
sales = [100, 150, 120, 200]

plt.figure(figsize=(10, 6))
plt.plot(months, sales, marker='o', linewidth=2)
plt.title('Monthly Sales')
plt.xlabel('Month')
plt.ylabel('Sales ($)')
plt.grid(True)
plt.savefig('sales_plot.png')
plt.show()
```

#### Where Used in Project
**In `naive_bayes_classifier.py`**:
```python
import matplotlib.pyplot as plt

def _plot_confusion_matrix(self, cm):
    plt.figure(figsize=(8, 6))
    # Plot confusion matrix
    plt.title('Confusion Matrix - Spam Detection')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.show()
```

#### Installation
```bash
pip install matplotlib
```

---

### 6. **Seaborn** 🎨

#### What is it?
A library built on **Matplotlib** for **statistical visualizations** - makes beautiful plots easier.

#### Why We Use It
- Creates professional-looking heatmaps
- Better color palettes
- Statistical plotting made simple

#### Code Example
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Create confusion matrix heatmap
cm = [[952, 10],
      [5, 158]]

plt.figure(figsize=(8, 6))
sns.heatmap(cm, 
            annot=True,           # Show numbers in cells
            fmt='d',              # Integer format
            cmap='Blues',         # Blue color scheme
            xticklabels=['Ham', 'Spam'],
            yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
```

#### Confusion Matrix Heatmap Output
```
                Predicted
             Ham    Spam
Actual  Ham  952     10
        Spam  5     158
```

**Color Explanation**:
- **Dark Blue** (952, 158) = Correct predictions
- **Light Blue** (10, 5) = Wrong predictions

#### Where Used in Project
**In `naive_bayes_classifier.py`**:
```python
import seaborn as sns

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham', 'Spam'],
            yticklabels=['Ham', 'Spam'])
```

#### Installation
```bash
pip install seaborn
```

---

## 📋 Dependency Summary Table

| Library | Version | Purpose | Used In | Installation |
|---------|---------|---------|---------|--------------|
| **NumPy** | ≥1.19.0 | Numerical computing | Background | `pip install numpy` |
| **Pandas** | ≥1.1.0 | Data manipulation | data_preparation.py | `pip install pandas` |
| **Scikit-learn** | ≥0.23.0 | Machine learning | feature_extraction.py, naive_bayes_classifier.py | `pip install scikit-learn` |
| **NLTK** | ≥3.5 | NLP preprocessing | data_preparation.py | `pip install nltk` |
| **Matplotlib** | ≥3.3.0 | Data visualization | naive_bayes_classifier.py | `pip install matplotlib` |
| **Seaborn** | ≥0.11.0 | Statistical plots | naive_bayes_classifier.py | `pip install seaborn` |

---

## 🔧 Installation Methods

### Method 1: Install Individually
```bash
pip install numpy
pip install pandas
pip install scikit-learn
pip install nltk
pip install matplotlib
pip install seaborn
```

### Method 2: Install All at Once
```bash
pip install numpy pandas scikit-learn nltk matplotlib seaborn
```

### Method 3: Using requirements.txt
Create file named `requirements.txt`:
```
numpy>=1.19.0
pandas>=1.1.0
scikit-learn>=0.23.0
nltk>=3.5
matplotlib>=3.3.0
seaborn>=0.11.0
```

Then install:
```bash
pip install -r requirements.txt
```

### Method 4: Using Anaconda (Alternative)
```bash
conda install numpy pandas scikit-learn nltk matplotlib seaborn
```

---

## 🎯 How Libraries Work Together

### Complete Data Flow with Libraries

```
📥 INPUT: Raw CSV File (spam.csv)
    ↓
📦 PANDAS: Load and read CSV
    df = pd.read_csv('spam.csv')
    ↓
🔤 NLTK: Clean and preprocess text
    - Tokenize with word_tokenize()
    - Remove stopwords
    - Stem with PorterStemmer()
    ↓
➗ PANDAS: Split into train/test
    train_test_split() from scikit-learn
    ↓
🔢 SCIKIT-LEARN: Extract TF-IDF features
    TfidfVectorizer transforms text → numbers
    ↓
🤖 SCIKIT-LEARN: Train Naive Bayes
    MultinomialNB() learns patterns
    ↓
📊 SCIKIT-LEARN: Evaluate model
    Metrics: accuracy, precision, recall
    ↓
📈 MATPLOTLIB + SEABORN: Visualize results
    Create confusion matrix heatmap
    ↓
📤 OUTPUT: Predictions on new messages
```

---

## ⚠️ Common Installation Issues & Solutions

### Issue 1: "ModuleNotFoundError: No module named 'sklearn'"
```bash
# Solution
pip install --upgrade scikit-learn
```

### Issue 2: "NLTK data not found"
```python
# Solution: Run in Python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### Issue 3: "Version conflict"
```bash
# Solution: Upgrade pip first
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Quick Verification

To verify all libraries are installed correctly:

```python
# test_imports.py
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

print("✓ NumPy version:", np.__version__)
print("✓ Pandas version:", pd.__version__)
print("✓ Scikit-learn version:", sklearn.__version__)
print("✓ NLTK version:", nltk.__version__)
print("✓ Matplotlib version:", plt.matplotlib.__version__)
print("✓ Seaborn version:", sns.__version__)
print("\n✅ All libraries installed successfully!")
```

Run:
```bash
python test_imports.py
```

---

## 💡 Key Points to Remember

1. **NumPy** = Foundation for numerical work
2. **Pandas** = Data loading and manipulation
3. **Scikit-learn** = Machine learning algorithms and tools
4. **NLTK** = Text preprocessing and cleaning
5. **Matplotlib** = Basic visualizations
6. **Seaborn** = Beautiful statistical plots

---

## 📚 Further Learning

- NumPy Tutorial: https://numpy.org/learn/
- Pandas Docs: https://pandas.pydata.org/docs/
- Scikit-learn Guide: https://scikit-learn.org/stable/
- NLTK Book: https://www.nltk.org/book/
- Matplotlib Examples: https://matplotlib.org/gallery.html
- Seaborn Tutorial: https://seaborn.pydata.org/tutorial.html

---

**This is PAGE 6 of the 10-page project guide explaining all dependencies and libraries used!** ✨
