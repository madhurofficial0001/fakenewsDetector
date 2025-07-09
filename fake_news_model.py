import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, request, render_template

# Initialize Flask app
app = Flask(__name__)

# Download NLTK stopwords if needed
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    text = re.sub(r"http\S+|www\S+|[^a-zA-Z\s]", '', text)  # Remove URLs and punctuation
    text = text.lower()
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# Load dataset
df = pd.read_csv("fake_or_real_news.csv")  # Make sure this file has a 'label' column
print(df.head())
print(df['label'].value_counts())

# Preprocess text column
df['cleaned_text'] = df['text'].apply(preprocess)

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_text'])

# Encode labels
y = df['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = PassiveAggressiveClassifier(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

def predict_news(news_text):
    clean = preprocess(news_text)
    vec = vectorizer.transform([clean])
    result = model.predict(vec)[0]
    return "REAL" if result == "REAL" else "FAKE"

# Flask route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    result = predict_news(news)
    return render_template('result.html', prediction=result)

# Example: test the function (this line runs only if you run script directly)
if __name__ == "__main__":
    sample_news = "This just in: aliens landed in New York!"
    print(predict_news(sample_news))
    app.run(debug=True)
