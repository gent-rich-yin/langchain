import datetime

from pydantic import BaseModel
from typing_extensions import TypedDict


class Position(TypedDict):
    date: datetime.date
    quantity: int

class BookPositions(BaseModel):
    book: str
    positions: list[Position]