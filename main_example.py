from fastapi import FastAPI, HTTPException
from schemas_example import GenreURLChoices, BandBase, BandCreate, BandWithID

# set --port argument, can't use 8000, the default uvicorn
# use localhost:{port} in browser
# use localhost:{port}/docs to look at the interactive, automatically created documentation

app = FastAPI()


BANDS = [
    {"id": 1, "name": "The Kinks", "genre": "Rock"},
    {"id": 2, "name": "Aphex Twin", "genre": "Electronic"},
    {"id": 3, "name": "Slowdive", "genre": "Shoegaze"},
    {"id": 4, "name": "Wu-Tang Clan", "genre": "Hip-Hop"},
    {
        "id": 5,
        "name": "Black Sabbath",
        "genre": "Metal",
        "albums": [{"title": "Master of Reality", "release_date": "1971-07-21"}],
    },
]


@app.get("/bands")
async def bands(
    genre: GenreURLChoices | None = None, has_albums: bool = False
) -> list[BandWithID]:
    band_list = [BandWithID(**b) for b in BANDS]

    if genre:
        band_list = [b for b in band_list if b.genre.value.lower() == genre.value]

    if has_albums:
        band_list = [b for b in band_list if len(b.albums) > 0]
    return band_list


@app.get("/bands/{band_id}")
async def band(band_id: int) -> BandWithID:
    band = next((BandWithID(**b) for b in BANDS if b["id"] == band_id), None)
    # Aaron: I'm a little confused, could we use `get` instead?

    if band is None:
        # status code 404
        raise HTTPException(status_code=404, detail="Band not found")
    return band


@app.get("/bands/genre/{genre}")
async def bands_for_genre(genre: GenreURLChoices) -> list[dict]:
    # originally this allowed any string to be used as an input and a list comprehension was used to find the value. However, this could result in server/computer waste, so we refactored to use a custom class that was restricted to a known set of genres
    return [b for b in BANDS if b["genre"].lower() == genre.value]


@app.post("/bands")
async def create_band(band_data: BandCreate) -> BandWithID:
    id = BANDS[-1]["id"] + 1
    band = BandWithID(id=id, **band_data.model_dump()).model_dump()
    BANDS.append(band)
    return band
