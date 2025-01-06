# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

class Item(BaseModel):
    id: int
    name: str
    description: str

# Sample data
items = [
    Item(id=1, name="Item 1", description="First item"),
    Item(id=2, name="Item 2", description="Second item")
]

@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.get("/items", response_model=List[Item])
def read_items():
    return items

@app.get("/items/{item_id}")
def read_item(item_id: int):
    item = next((item for item in items if item.id == item_id), None)
    return item

@app.post("/items")
def create_item(item: Item):
    items.append(item)
    return item