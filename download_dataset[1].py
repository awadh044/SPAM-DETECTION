import urllib.request
import os

def download_spam_dataset():
    """Download SMS Spam Collection dataset"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    filename = "smsspamcollection.zip"
    
    if not os.path.exists('spam.csv'):
        print("Downloading dataset...")
        urllib.request.urlretrieve(url, filename)
        
        # Extract and rename
        import zipfile
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall()
        
        os.rename('SMSSpamCollection', 'spam.csv')
        os.remove(filename)
        print("Dataset ready!")
    else:
        print("Dataset already exists!")

if __name__ == "__main__":
    download_spam_dataset()