from pydantic import BaseModel

class Employee(BaseModel):
    name: str
    employee_id: str

class RegistrationRequest(BaseModel):
    employee: Employee