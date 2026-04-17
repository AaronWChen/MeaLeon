"""
Tests for the search service.

Run with: pytest search_service/tests/ -v

Uses httpx.AsyncClient with ASGITransport to test the FastAPI app
without spinning up a real server. Redis is mocked so tests are
hermetic and fast.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.models import RecipeSearchResult, SearchRequest, SearchResponse
from app.edamam_client import EdamamClient, EdamamRateLimitError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_EDAMAM_HIT = {
    "recipe": {
        "uri": "http://www.edamam.com/ontologies/edamam.owl#recipe_abc123",
        "label": "Classic Lasagna",
        "source": "Epicurious",
        "url": "https://www.epicurious.com/recipes/classic-lasagna",
        "cuisineType": ["italian"],
        "ingredientLines": [
            "2 cups all-purpose flour",
            "3 large eggs",
            "1 lb ground beef",
            "2 cups tomato sauce",
            "1 cup ricotta cheese",
            "2 cups shredded mozzarella",
        ],
        "calories": 450.0,
        "totalTime": 90,
        "dietLabels": ["High-Protein"],
        "healthLabels": ["Egg-Free"],
        "images": {
            "REGULAR": {
                "url": "https://example.com/lasagna.jpg",
                "width": 300,
                "height": 300,
            }
        },
    }
}


@pytest.fixture
def mock_redis():
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)  # cache miss by default
    redis.setex = AsyncMock(return_value=True)
    redis.keys = AsyncMock(return_value=[])
    redis.delete = AsyncMock(return_value=0)
    redis.aclose = AsyncMock()
    return redis


@pytest.fixture
def mock_edamam():
    client = AsyncMock(spec=EdamamClient)
    client.search = AsyncMock(
        return_value=[
            RecipeSearchResult(
                edamam_id="http://www.edamam.com/ontologies/edamam.owl#recipe_abc123",
                label="Classic Lasagna",
                source="Epicurious",
                url="https://www.epicurious.com/recipes/classic-lasagna",
                cuisine_types=["italian"],
                ingredient_lines=["2 cups flour", "3 eggs"],
                ingredient_names=["flour", "eggs"],
            )
        ]
    )
    return client


# ---------------------------------------------------------------------------
# EdamamClient unit tests
# ---------------------------------------------------------------------------


class TestEdamamClient:
    def test_parse_ingredient_name_strips_quantity_and_unit(self):
        client = EdamamClient.__new__(EdamamClient)
        assert client._parse_ingredient_name("2 cups all-purpose flour") == "flour"

    def test_parse_ingredient_name_handles_fractions(self):
        client = EdamamClient.__new__(EdamamClient)
        assert (
            client._parse_ingredient_name("1/2 tsp freshly ground pepper") == "pepper"
        )

    def test_parse_ingredient_name_strips_prep_words(self):
        client = EdamamClient.__new__(EdamamClient)
        result = client._parse_ingredient_name("3 cloves garlic, minced")
        assert "garlic" in result

    def test_parse_ingredient_name_multi_word(self):
        client = EdamamClient.__new__(EdamamClient)
        result = client._parse_ingredient_name("1 tbsp olive oil")
        assert "olive oil" in result

    def test_best_image_prefers_large(self):
        client = EdamamClient.__new__(EdamamClient)
        images = {
            "THUMBNAIL": {"url": "small.jpg"},
            "REGULAR": {"url": "medium.jpg"},
            "LARGE": {"url": "large.jpg"},
        }
        assert client._best_image(images) == "large.jpg"

    def test_best_image_falls_back(self):
        client = EdamamClient.__new__(EdamamClient)
        images = {"THUMBNAIL": {"url": "small.jpg"}}
        assert client._best_image(images) == "small.jpg"

    def test_best_image_empty(self):
        client = EdamamClient.__new__(EdamamClient)
        assert client._best_image({}) is None

    @pytest.mark.asyncio
    async def test_search_parses_hit(self, respx_mock):
        """Integration-style: real EdamamClient with mocked HTTP."""
        import respx
        from httpx import Response

        client = EdamamClient(app_id="test_id", app_key="test_key")
        respx_mock.get(url__startswith="https://api.edamam.com").mock(
            return_value=Response(200, json={"hits": [MOCK_EDAMAM_HIT]})
        )
        results = await client.search("lasagna", cuisine="italian")
        assert len(results) == 1
        assert results[0].label == "Classic Lasagna"
        assert "flour" in results[0].ingredient_names


# ---------------------------------------------------------------------------
# API route tests
# ---------------------------------------------------------------------------


class TestSearchRoutes:

    @pytest.mark.asyncio
    async def test_health(self, mock_redis, mock_edamam):
        app.state.redis = mock_redis
        app.state.edamam = mock_edamam
        from app.cache import SearchCache

        app.state.cache = SearchCache(mock_redis)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            resp = await ac.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_search_post_cache_miss(self, mock_redis, mock_edamam):
        app.state.redis = mock_redis
        app.state.edamam = mock_edamam
        from app.cache import SearchCache

        app.state.cache = SearchCache(mock_redis)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            resp = await ac.post(
                "/search", json={"dish_name": "lasagna", "cuisine": "italian"}
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["dish_name"] == "lasagna"
        assert data["from_cache"] is False
        assert len(data["recipes"]) > 0

    @pytest.mark.asyncio
    async def test_search_get_convenience(self, mock_redis, mock_edamam):
        app.state.redis = mock_redis
        app.state.edamam = mock_edamam
        from app.cache import SearchCache

        app.state.cache = SearchCache(mock_redis)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            resp = await ac.get("/search?dish_name=lasagna&cuisine=italian")

        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_search_returns_cached_on_hit(self, mock_redis, mock_edamam):
        cached_response = SearchResponse(
            dish_name="lasagna",
            cuisine="italian",
            recipes=[],
            from_cache=True,
        )
        mock_redis.get = AsyncMock(return_value=cached_response.model_dump_json())

        app.state.redis = mock_redis
        app.state.edamam = mock_edamam
        from app.cache import SearchCache

        app.state.cache = SearchCache(mock_redis)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            resp = await ac.post(
                "/search", json={"dish_name": "lasagna", "cuisine": "italian"}
            )

        assert resp.status_code == 200
        assert resp.json()["from_cache"] is True
        # Edamam should NOT have been called
        mock_edamam.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_502_on_edamam_error(self, mock_redis, mock_edamam):
        mock_edamam.search.side_effect = Exception("API down")
        app.state.redis = mock_redis
        app.state.edamam = mock_edamam
        from app.cache import SearchCache

        app.state.cache = SearchCache(mock_redis)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            resp = await ac.post("/search", json={"dish_name": "lasagna"})

        assert resp.status_code == 502

    def test_search_request_normalises_input(self):
        req = SearchRequest(dish_name="  Lasagna  ", cuisine="  Italian  ")
        assert req.dish_name == "lasagna"
        assert req.cuisine == "italian"

    def test_search_response_all_ingredients_deduplicates(self):
        r1 = RecipeSearchResult(
            edamam_id="1",
            label="A",
            source="S",
            url="u",
            ingredient_names=["garlic", "onion"],
        )
        r2 = RecipeSearchResult(
            edamam_id="2",
            label="B",
            source="S",
            url="u",
            ingredient_names=["garlic", "tomato"],  # garlic is a duplicate
        )
        response = SearchResponse(dish_name="test", cuisine="", recipes=[r1, r2])
        all_ingr = response.all_ingredients
        assert all_ingr.count("garlic") == 1
        assert "onion" in all_ingr
        assert "tomato" in all_ingr
