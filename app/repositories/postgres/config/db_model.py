from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List
from datetime import datetime
from decimal import Decimal


class VendorAccount(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    name: str = Field(index=True, max_length=255)
    description: Optional[str] = Field(default=None, max_length=1024)
    location: Optional[str] = Field(default=None, max_length=255)
    email: str = Field(index=True, unique=True, max_length=255)
    hashed_password: str = Field(max_length=255)
    created_at: datetime
    updated_at: Optional[datetime] = None

    tags: List["VendorTag"] = Relationship(back_populates="vendor")

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "location": self.location,
            "description": self.description,
            "tags": [tag.tag_name for tag in self.tags] if self.tags else [],
        }
    

class VendorTag(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    vendor_email: str = Field(foreign_key="vendoraccount.email", index=True, max_length=255)
    tag_name: str = Field(index=True, max_length=50)

    vendor: Optional[VendorAccount] = Relationship(back_populates="tags")


class RequestProcurement(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str = Field(index=True, max_length=255)
    category: str = Field(index=True, max_length=100)
    description: Optional[str] = None
    requirements: Optional[str] = None
    price: Decimal = Field(decimal_places=2, max_digits=20)
    location: Optional[str] = Field(default=None, max_length=255)
    due_date: datetime
    status: str = Field(index=True, max_length=50)
    created_at: datetime
    updated_at: Optional[datetime] = None

    tags: List["RequestTag"] = Relationship(back_populates="request")

    def to_dict(self):
        return {
                "id": self.id,
                "title": self.title,
                "category": self.category,
                "description": self.description,
                "requirements": self.requirements,
                "price": str(self.price),  # Decimal needs str for JSON-safe output
                "location": self.location,
                "due_date": self.due_date.isoformat(),
                "status": self.status,
                "created_at": self.created_at.isoformat(),
                "updated_at": self.updated_at.isoformat() if self.updated_at else None,
                "tags": [tag.tag_name for tag in self.tags] if self.tags else [],  # only keep tag name
            }



class RequestTag(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    request_id: int = Field(foreign_key="requestprocurement.id", index=True)
    tag_name: str = Field(index=True, max_length=50)

    request: Optional[RequestProcurement] = Relationship(back_populates="tags")


class AdminAccount(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True, max_length=150)
    hashed_password: str = Field(max_length=255)
    total_budget: Decimal = Field(default=0, decimal_places=2, max_digits=20)
    created_at: datetime
    updated_at: Optional[datetime] = None
