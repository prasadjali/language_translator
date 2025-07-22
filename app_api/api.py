
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import timeit
import json
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Body, Depends
from fastapi.encoders import jsonable_encoder
#from language_translator import __version__ as model_version
#from trans import init_model, translate
from app_api.model import mymodel
src_lang, tgt_lang = "eng_Latn", "hin_Deva"
#src_lang, tgt_lang = "eng_Latn", "kan_Knda"
#src_lang, tgt_lang = "eng_Latn", "tam_Taml"

from app_api import __version__, schemas
from app_api.schemas.health import Health
from app_api.schemas.predict import PredictionResults, MultipleDataInputs
from app_api.config import settings
from app_api.trans import get_model, get_tokenizer, get_ip


api_router = APIRouter()


@api_router.get("/health", response_model=Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    )

    return health.dict()



example_input = {
    "inputs": [
        {
            "FarmerId": 79,
            "TgtLang": "hin_Deva",
            "Text": """
            "Disease Name": "Powdery Mildew",
            "Local Name": "Dhan Ki Kharash",
            "Disease Description": "Powdery mildew is a fungal disease that 
            affects a variety of plants, including rice. It is characterized by the presence
            of white, powdery fungal growth on the leaves, stems, and sometimes flowers. 
            The disease thrives in warm, dry conditions and can lead 
            to reduced photosynthesis, stunted growth, 
            and lower yields if not managed properly."
            """,
          }
    ]
}


@api_router.post("/predict", response_model=PredictionResults, status_code=200)
async def predict(input_data: MultipleDataInputs = Body(..., example=example_input)) -> Any:
    """
    Language Translation Endpoint
    """

    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))
    
    # Import the model, tokenizer, and ip from main if they are initialized there

    model = mymodel.get_model()
    print(model)
    tokenizer = mymodel.get_tokenizer()
    ip = mymodel.get_ip()
    if model is None or tokenizer is None or ip is None:
        raise HTTPException(status_code=500, detail="Model, tokenizer, or IndicProcessor not initialized.")

    if input_df.empty:
        raise HTTPException(status_code=400, detail="Input data is empty.")
    if 'TgtLang' not in input_df.columns or 'Text' not in input_df.columns:
        raise HTTPException(status_code=400, detail="Input data must contain 'TgtLang' and 'Text' columns.")
    if input_df['TgtLang'].isnull().any() or input_df['Text'].isnull().any():
        raise HTTPException(status_code=400, detail="Input data must not contain null values in 'TgtLang' or 'Text' columns.")
        
    
    start = timeit.default_timer()
    response = mymodel.translate(
        input_sentences=input_df['Text'].tolist(),
        src_lang=src_lang,
        tgt_lang=input_df['TgtLang'].iloc[0],
        model=model,
        tokenizer=tokenizer,
        ip=ip
    )
    end = timeit.default_timer()
    print(f"Time taken for translation: {end - start} seconds")
    
    if response is None:
        raise HTTPException(status_code=400, detail=json.loads("errors"))

    retstr = "".join(response)
    response = retstr.replace("<pad>", "").replace("<s>", "").replace("</s>","").strip()
    result = {
        "TgtLang": input_df['TgtLang'].iloc[0],
        "Text": response,
        "errors": None,
        "version": __version__
    }
    print(f"Response: {input_df['FarmerId'].iloc[0]}")
    if input_df['FarmerId'].iloc[0] is not None:
        result["FarmerId"] = input_df['FarmerId'].iloc[0]
    return PredictionResults(**result)

