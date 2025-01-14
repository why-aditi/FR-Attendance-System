from pydantic import BaseModel
from datetime import datetime

class User(BaseModel):
    name: str
    id: str
    clocked_in: bool
    clock_in_time: datetime

class RegistrationRequest(BaseModel):
    name: str
    employee_id: str

class RegistrationResponse(BaseModel):
    status: str
    message: str

class ProcessingResponse(BaseModel):
    status: str
    embedding_path: str