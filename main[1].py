from data_preparation import DataPreparation
from feature_extraction import FeatureExtraction
from naive_bayes_classifier import SpamDetectionModel

def main():
    print("=" * 50)
    print("SPAM DETECTION USING NAIVE BAYES")
    print("=" * 50)
    
    # Step 1: Load and preprocess data
    print("\n[Step 1] Loading and preprocessing data...")
    dp = DataPreparation()
    df = dp.load_data('spam.csv')  # Download SMS Spam Collection dataset
    df = dp.preprocess_data(df)
    
    # Step 2: Split data
    print("\n[Step 2] Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = dp.split_data(df)
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Step 3: Feature extraction
    print("\n[Step 3] Extracting features...")
    fe = FeatureExtraction()
    X_train_tfidf = fe.fit_transform(X_train)
    X_test_tfidf = fe.transform(X_test)
    print(f"Feature matrix shape: {X_train_tfidf.shape}")
    
    # Step 4: Train model
    print("\n[Step 4] Training Naive Bayes model...")
    model = SpamDetectionModel()
    model.train(X_train_tfidf, y_train)
    
    # Step 5: Evaluate model
    print("\n[Step 5] Evaluating model...")
    metrics = model.evaluate(X_test_tfidf, y_test)
    
    # Step 6: Test on custom messages
    print("\n[Step 6] Testing on custom messages...")
    test_messages = [
        "You have won a free iPhone. Click here to claim your prize!",
        "Hi, how are you doing today?",
        "Congratulations! You've been selected for a special offer.",
        "Let's meet tomorrow at 3pm for coffee"
    ]
    
    for msg in test_messages:
        cleaned_msg = dp.clean_text(msg)
        model.predict_single_message(cleaned_msg, fe)

if __name__ == "__main__":
    main()