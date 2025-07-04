from pydantic import BaseModel, EmailStr
from typing import Optional, List
from decimal import Decimal
from datetime import datetime
from sqlmodel import SQLModel

class RequestProcurementCreate(BaseModel):
    title: str
    category: str
    description: Optional[str] = None
    requirements: Optional[str] = None
    price: Decimal
    location: Optional[str] = None
    due_date: datetime
    tags: List[str] 

class VendorRegister(BaseModel):
    name: str
    description: Optional[str] = None
    location: Optional[str] = None
    email: EmailStr
    password: str
    tags: Optional[List[str]] = [] 


class VendorUpdate(BaseModel):
    name: str
    description: Optional[str] = None
    location: Optional[str] = None
    email: EmailStr
    tags: Optional[List[str]] = [] 

class VendorListRequest(BaseModel):
    page: Optional[int] = 1
    limit: Optional[int] = 10
    
class VendorTagRead(SQLModel):
    id: int
    tag_name: str

class VendorAccountRead(SQLModel):
    id: int
    name: str
    description: Optional[str] = None
    location: Optional[str] = None
    email: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    tags: List[VendorTagRead] = []
    

class VendorListResponse(BaseModel):
    resp_msg: str
    resp_data: List[VendorAccountRead]

class VendorDetailResponse(BaseModel):
    resp_msg: str
    resp_data: VendorAccountRead


class VendorLogin(BaseModel):
    email: EmailStr
    password: str

class RequestTagRead(SQLModel):
    id: int
    tag_name: str
    
class RequestProcurementRead(SQLModel):
    id: int
    title: str
    category: str
    description: Optional[str] = None
    requirements: Optional[str] = None
    price: Decimal
    location: Optional[str] = None
    due_date: datetime
    status: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    tags: List[RequestTagRead] = []

class RequestProcurementListRequest(BaseModel):
    page: Optional[int] = 1
    limit: Optional[int] = 10

class ProcListItems(BaseModel):
    total_count: int
    vendors: List[RequestProcurementRead]

class RequestProcurementListResponse(BaseModel):
    resp_msg: str
    resp_data: ProcListItems