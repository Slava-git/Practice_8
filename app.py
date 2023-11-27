from fastapi import FastAPI, Request, HTTPException

from src.inference import Inference

app = FastAPI()
inference_model = Inference()

@app.post("/predict")
async def predict(request: Request):
    try:
        request_data = await request.json()

        text = request_data.get("text")

        predictions = inference_model(text)
        return {"predictions": predictions.tolist()}
    except Exception as e:

        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    
@app.get("/")
async def root():
    return {"message": "NLP Inference API"}