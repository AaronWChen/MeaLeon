from enum import Enum
from fastapi import FastAPI, HTTPException

# use localhost:{port} in browser
# use localhost:{port}/docs to look at the interactive, automatically created documentation

app = FastAPI()


class GenreURLChoices(Enum):
    ROCK = "rock"
    ELECTRONIC = "electronic"
    METAL = "metal"
    HIP_HOP = "hip-hop"
    SHOEGAZE = "shoegaze"


BANDS = [
    {"id": 1, "name": "The Kinks", "genre": "Rock"},
    {"id": 2, "name": "Aphex Twin", "genre": "Electronic"},
    {"id": 3, "name": "Slowdive", "genre": "Shoegaze"},
    {"id": 4, "name": "Wu-Tang Clan", "genre": "Hip-Hop"},
    {"id": 5, "name": "Black Sabbath", "genre": "Metal"},
]


@app.get("/bands")
async def bands() -> list[dict]:
    return BANDS


@app.get("/bands/{band_id}")
async def band(band_id: int) -> dict:
    band = next((b for b in BANDS if b["id"] == band_id), None)
    # Aaron: I'm a little confused, could we use `get` instead?

    if band is None:
        # status code 404
        raise HTTPException(status_code=404, detail="Band not found")
    return band


@app.get("/bands/genre/{genre}")
async def bands_for_genre(genre: GenreURLChoices) -> list[dict]:
    # originally this allowed any string to be used as an input and a list comprehension was used to find the value. However, this could result in server/computer waste, so we refactored to use a custom class that was restricted to a known set of genres
    return [b for b in BANDS if b["genre"].lower() == genre.value]
