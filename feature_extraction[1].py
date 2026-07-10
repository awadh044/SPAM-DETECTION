from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureExtraction:
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2)  # unigrams and bigrams
        )
    
    def fit_transform(self, X_train):
        """Fit vectorizer on training data"""
        return self.vectorizer.fit_transform(X_train)
    
    def transform(self, X_test):
        """Transform test data"""
        return self.vectorizer.transform(X_test)
    
    def get_feature_names(self):
        """Get feature names"""
        return self.vectorizer.get_feature_names_out()