# BugBytes instructor likes to use the schemas.py file and add all the Pydantic schemas to that file
from datetime import date
from enum import Enum
from pydantic import BaseModel


class GenreURLChoices(Enum):
    ROCK = "rock"
    ELECTRONIC = "electronic"
    METAL = "metal"
    HIP_HOP = "hip-hop"
    SHOEGAZE = "shoegaze"


class Album(BaseModel):
    title: str
    release_date: date


class BandBase(BaseModel):
    name: str
    genre: str
    albums: list[Album] = []


class BandCreate(BandBase):
    # only pass because it is strictly inheriting and not adding other fields
    pass


class BandWithID(BandBase):
    id: int
