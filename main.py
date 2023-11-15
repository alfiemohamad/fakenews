

# from predict import tmodelsvm

# testsvm = 'gibran berpasangan dengan anies baswedan di pilpres 2024'
# presvm = tmodelsvm.predict([testsvm])
# presvm



from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from predict import tmodelsvm

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class InputData(BaseModel):
    text: str

@app.post("/predict")
async def predict_text(input_data: InputData):
    try:
        text_to_predict = input_data.text
        prediction = tmodelsvm.predict([text_to_predict])
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)