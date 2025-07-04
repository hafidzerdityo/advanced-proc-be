from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime

class ResponseDefault(BaseModel):
    resp_msg: str 
    resp_data: dict