from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns

class SpamDetectionModel:
    def __init__(self):
        self.model = MultinomialNB()
        self.is_trained = False
    
    def train(self, X_train, y_train):
        """Train the Naive Bayes model"""
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("Model trained successfully!")
    
    def predict(self, X_test):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        """Get prediction probabilities"""
        return self.model.predict_proba(X_test)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print("=" * 50)
        print("MODEL PERFORMANCE METRICS")
        print("=" * 50)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print("\n" + classification_report(y_test, y_pred, 
                                          target_names=['Ham', 'Spam']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        self._plot_confusion_matrix(cm)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }
    
    def _plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Ham', 'Spam'],
                    yticklabels=['Ham', 'Spam'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.show()
        
    def predict_single_message(self, message, vectorizer):
        """Predict for a single message"""
        X = vectorizer.transform([message])
        prediction = self.predict(X)[0]
        probability = self.predict_proba(X)[0]
        
        label = "SPAM" if prediction == 1 else "HAM"
        confidence = probability[prediction] * 100
        
        print(f"\nMessage: {message}")
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"Ham probability: {probability[0]*100:.2f}%")
        print(f"Spam probability: {probability[1]*100:.2f}%")