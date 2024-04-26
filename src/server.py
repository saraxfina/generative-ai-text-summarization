from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from fastapi import FastAPI
import uvicorn
import nest_asyncio
import json

app = FastAPI()

model = AutoModelForSeq2SeqLM.from_pretrained('../models/fine_tuned/fine_tuned_model_v2.keras', from_tf=True)
tokenizer = AutoTokenizer.from_pretrained('../models/fine_tuned/fine_tuned_model_v2.keras', from_tf=True)

@app.post('/predict')
async def predict(data: dict):
    text = data["input_text"]
    encoded_text = tokenizer.encode(text[:2844], return_tensors="pt")
    summary_ids = model.generate(encoded_text, max_length=60)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return {"summary": summary }

@app.get("/")
async def root():
    return {"message": "Usage: Input: A news article. Output: An article summary."}

@app.get("/authors")
async def authors():
    return {"message": "Scott Mayer - Sarafina Gonzalez"}

# run this script to start the FastAPI service
if __name__ == "__main__":
    nest_asyncio.apply()
    uvicorn.run(app)
