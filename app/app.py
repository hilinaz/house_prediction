from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd
from utils.preprocessing import preprocess_data_predict  # updated import

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# Load the trained model once
model = joblib.load('model/model.pkl')

@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("form.html", {"request": request, "prediction": None})

@app.post("/", response_class=HTMLResponse)
async def form_post(
    request: Request,
    area: float = Form(...),
    bedrooms: int = Form(...),
    bathrooms: int = Form(...),
    year_built: int = Form(...)
):
    # Prepare input data from form
    input_data = {
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'year_built': year_built
    }
    input_df = pd.DataFrame([input_data])
    X = preprocess_data_predict(input_df)  # updated function call
    prediction = model.predict(X)[0]
    prediction = round(prediction, 2)

    return templates.TemplateResponse("form.html", {"request": request, "prediction": prediction})
