from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Query(BaseModel):
    text: str

def preprocess_query(query_text):
    # Implement preprocessing logic here
    preprocessed_text = query_text.lower()  # Example preprocessing
    return preprocessed_text

@app.post("/query")
async def handle_query(query: Query):
    preprocessed_text = preprocess_query(query.text)
    # Proceed to the retrieval layer with the preprocessed query
    return {"preprocessed_query": preprocessed_text}