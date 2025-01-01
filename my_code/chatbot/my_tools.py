import datetime
from typing import Annotated

from chatbot.Position import BookPositions
from langchain_core.tools import tool

@tool
def get_positions(book: Annotated[str, 'financial book that holds the position']) -> Annotated[BookPositions, 'historical position for the book']:
    """Retrieve historical position for the book"""
    positions=[{
            "date": datetime.date(2024, 10, 1),
            "quantity": 11234
        }, {
            "date": datetime.date(2024, 11, 1),
            "quantity": 15000
        }, {
            "date": datetime.date(2024, 12, 1),
            "quantity": 10000
        }]
    return BookPositions(book=book, positions=positions)
