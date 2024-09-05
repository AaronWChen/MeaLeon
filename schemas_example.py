# BugBytes instructor likes to use the schemas.py file and add all the Pydantic schemas to that file
from datetime import date
from enum import Enum
from pydantic import BaseModel, field_validator


class GenreURLChoices(Enum):
    ROCK = "rock"
    ELECTRONIC = "electronic"
    METAL = "metal"
    HIP_HOP = "hip-hop"
    SHOEGAZE = "shoegaze"


class GenreChoices(Enum):
    ROCK = "Rock"
    ELECTRONIC = "Electronic"
    METAL = "Metal"
    HIP_HOP = "Hip-Hop"
    SHOEGAZE = "Shoegaze"


class Album(BaseModel):
    title: str
    release_date: date


class BandBase(BaseModel):
    name: str
    genre: GenreChoices
    albums: list[Album] = []


class BandCreate(BandBase):
    # only pass because it is strictly inheriting and not adding other fields
    @field_validator("genre", mode="before")
    def title_case_genre(cls, value):
        return value.title()


class BandWithID(BandBase):
    id: int
