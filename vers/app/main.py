from fastapi import FastAPI, File, Response, UploadFile

app = FastAPI(title="Box of crayons")


@app.get(
    "/",
)
def get_predictions(file: UploadFile = File(...)):
    return {"response": "Hello"}
