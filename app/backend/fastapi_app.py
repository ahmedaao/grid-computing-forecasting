# Import packages
import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

# Import src modules
from grid_computing_forecasting import config, dataset
from grid_computing_forecasting.modeling import predict


# Load environment variables from .env file
load_dotenv()

# Load pickle object for test API
df = dataset.load_pickle_file(
    os.path.join(config.root_dir, "app/backend/df_sample.pkl")
)


app = FastAPI(title="MyApp", description="Grid Computing Forecasting")


class Item(BaseModel):
    job_id: int


@app.post("/prediction")
def grownet_prediction(request: Item):
    job_id = request.job_id

    result = predict.with_xgboost(config.root_dir, "models/xgboost.pkl", df.loc[job_id])

    return {"prediction": float(result)}
