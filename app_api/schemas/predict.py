from typing import Any, List, Optional

from pydantic import BaseModel


class PredictionResults(BaseModel):
    FarmerId: Optional[int] = None
    TgtLang: Optional[str]
    Text: Optional[str]
    errors: Optional[Any]
    version: str
    class Config:
        orm_mode = True
        exclude_none = True

class DataInputSchema(BaseModel):
    FarmerId: Optional[int] = None
    TgtLang: Optional[str]
    Text: Optional[str]
    
class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

