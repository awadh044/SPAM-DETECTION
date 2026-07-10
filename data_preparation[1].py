import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class DataPreparation:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def load_data(self, filepath):
        """Load spam dataset"""
        # Using SMS Spam Collection dataset format
        df = pd.read_csv(filepath, sep='\t', header=None, names=['label', 'message'])
        print(f"Dataset shape: {df.shape}")
        print(f"Class distribution:\n{df['label'].value_counts()}")
        return df
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and non-alphabetic tokens
        tokens = [self.stemmer.stem(token) for token in tokens 
                  if token.isalpha() and token not in self.stop_words]
        
        return ' '.join(tokens)
    
    def preprocess_data(self, df):
        """Preprocess the entire dataset"""
        df['cleaned_message'] = df['message'].apply(self.clean_text)
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        return df
    
    def split_data(self, df, test_size=0.2, random_state=42):
        """Split into train and test sets"""
        X_train, X_test, y_train, y_test = train_test_split(
            df['cleaned_message'], 
            df['label'],
            test_size=test_size,
            random_state=random_state,
            stratify=df['label']
        )
        return X_train, X_test, y_train, y_test