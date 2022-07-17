from typing import Optional
from joblib import load
from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI()

vector = load("vectorizer.joblib")
model = load("model.joblib")

class get_sentiment(BaseModel):
    text :str


@app.get("/")
def read_root():
    return {"Sentiment Analysis": "Python"}


@app.post("/prediction")
def get_prediction(x:get_sentiment):
    text = [x.text]
    vec = vector.transform(text)
    prediction = model.predict(vec)
    if prediction==1:
        prediction="positive"
    else:
        prediction = "negative"

    return {"sentence" :x.text,"prediction":prediction}



