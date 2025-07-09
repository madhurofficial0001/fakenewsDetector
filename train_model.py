import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
import joblib

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Load dataset
df = pd.read_csv("fake_or_real_news.csv")

# Manually add label (if not already there)
# Uncomment only if your CSV doesn't have a 'label' column:
# df['label'] = df['subject'].apply(lambda x: 'FAKE' if x == 'fake' else 'REAL')

# Preprocess text
def preprocess(text):
    text = re.sub(r"http\S+|www\S+|[^a-zA-Z\s]", '', str(text))
    text = text.lower()
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

df['cleaned_text'] = df['text'].apply(preprocess)

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = PassiveAggressiveClassifier()
model.fit(X_train, y_train)

# Save model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model and vectorizer saved successfully!")
