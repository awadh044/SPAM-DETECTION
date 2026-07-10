# SPAM-DETECTION
A machine learning text classification system that automatically detects spam messages using Naive Bayes algorithm with 98.7% accuracy.

Key Highlights:
🎯 Algorithm: Multinomial Naive Bayes
📊 Dataset: 5,572 SMS messages
✅ Accuracy: 98.7%
⚡ Fast: Trains in < 1 second
🛡️ Real-world: Email/SMS filtering
What does it do?
Loads and cleans SMS data
Extracts text features (TF-IDF)
Trains Naive Bayes model
Evaluates performance
Predicts new messages with confidence scor
Example:
Code
Input: "You won FREE iPhone! Click NOW!!!"
→ Output: 🚨 SPAM (97.3% confidence)

Input: "Can we meet tomorrow?"
→ Output: ✓ HAM (99.8% confidence)

 😊Project Structure 

spam-detection-naive-bayes/

├── data_preparation.py

├── feature_extraction.py

├── naive_bayes_classifier.py

├── main.py

├── download_dataset.py

├── spam.csv

├── confusion_matrix.png

└── requirements.txt

• How Naive Bayes Works 

Bayes' Theorem: P(Spam|Message) = P(Message|Spam) × P(Spam) / P(Message)

Features: Words and their frequencies (TF-IDF scores)

Training: Learns probability of each word appearing in spam vs. ham

Prediction: Calculates probability message is spam based on its words
