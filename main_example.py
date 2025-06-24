from fastapi import FastAPI, HTTPException, Path, Query, Depends
from models_example import GenreURLChoices, BandBase, BandCreate, Band, Album
from sqlmodel import Session, select
from typing import Annotated
from contextlib import asynccontextmanager
from db_example import init_db, get_session


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


# set --port argument, can't use 8000, the default uvicorn
# use localhost:{port} in browser
# use localhost:{port}/docs to look at the interactive, automatically created documentation

app = FastAPI(lifespan=lifespan)


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
    genre: GenreURLChoices | None = None,
    # has_albums: bool = False,
    name_query: Annotated[str | None, Query(max_length=10)] = None,
    session: Session = Depends(get_session),
) -> list[Band]:
    band_list = session.exec(select(Band)).all()

    if genre:
        band_list = [b for b in band_list if b.genre.value.lower() == genre.value]

    # if has_albums:
    #     band_list = [b for b in band_list if len(b.albums) > 0]

    if name_query:
        band_list = [b for b in band_list if name_query.lower() in b.name.lower()]

    return band_list


@app.get("/bands/{band_id}")
async def band(
    band_id: Annotated[int, Path(title="The band ID")],
    session: Session = Depends(get_session),
) -> Band:
    band = session.get(Band, band_id)
    # Aaron: I'm a little confused, could we use `get` instead?

    if band is None:
        # status code 404
        raise HTTPException(status_code=404, detail="Band not found")
    return band


# @app.get("/bands/genre/{genre}")
# async def bands_for_genre(genre: GenreURLChoices) -> list[dict]:
#     # originally this allowed any string to be used as an input and a list comprehension was used to find the value. However, this could result in server/computer waste, so we refactored to use a custom class that was restricted to a known set of genres
#     return [b for b in BANDS if b["genre"].lower() == genre.value]


@app.post("/bands")
async def create_band(
    band_data: BandCreate, session: Session = Depends(get_session)
) -> Band:

    band = Band(name=band_data.name, genre=band_data.genre)
    session.add(band)

    if band_data.albums:
        for album in band_data.albums:
            album_obj = Album(
                title=album.title, release_date=album.release_date, band=band
            )
            session.add(album_obj)

    session.commit()
    session.refresh(band)

    return band
