from pydantic import BaseModel
from typing import List

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str
    success: bool
    references: List[str]
