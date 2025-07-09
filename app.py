from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

app = FastAPI()
origins = [
    "http://localhost:3000",  # your frontend origin, e.g., React dev server
    "http://localhost:5173",  # or Vite, etc.
    # add other allowed origins if needed
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def preprocess(text):
    text = re.sub(r"http\S+|www\S+|[^a-zA-Z\s]", '', str(text))
    text = text.lower()
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

@app.get("/")
def read_root():
    return {"message": "Fake News Detector API running."}

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    text = data.get("news", "")
    preprocessed = preprocess(text)
    vectorized = vectorizer.transform([preprocessed])
    prediction = model.predict(vectorized)[0]
    return {"prediction": prediction}

