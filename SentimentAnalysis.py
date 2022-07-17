import string
from pydantic import BaseModel

class SentimentAnalysis(BaseModel):
   float: string
