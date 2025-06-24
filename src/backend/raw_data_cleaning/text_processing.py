#!/usr/bin/env python3

# This script contains reusable methods to do recipe processing for MeaLeon
from hashlib import sha256
from urllib.parse import urlparse


def unique_name_from_str(string: str) -> str:
    """
    Generates a unique id name. Preferably, the input string is the netloc and path of the URL
    """
    return sha256(string.encode("utf8")).hexdigest()


def mealeon_id_assigner(origin_name: str, recipe_url: str) -> str:
    """
    Generates a unique id name from original URL pushed through the sha256 hash function
    """
    parsed_url = urlparse(recipe_url)
    sha_loc_path = unique_name_from_str(f"{parsed_url.netloc}{parsed_url.path}")

    return f"{origin_name}-{sha_loc_path}"


def mealeon_id_all_recipes_assigner(recipe_url: str) -> str:
    """Passthrough function to mealeon_id_assigner with 'AllRecipes' as the origin. Doing it this way because polars map_elements doesn't seem to take *args or **kwargs"""
    return mealeon_id_assigner("AllRecipes", recipe_url)


def mealeon_id_bbc_assigner(recipe_url: str) -> str:
    """Passthrough function to mealeon_id_assigner with 'BBC' as the origin. Doing it this way because polars map_elements doesn't seem to take *args or **kwargs"""
    return mealeon_id_assigner("BBC", recipe_url)


def mealeon_id_cookstr_assigner(recipe_url: str) -> str:
    """Passthrough function to mealeon_id_assigner with 'Cookstr' as the origin. Doing it this way because polars map_elements doesn't seem to take *args or **kwargs"""
    return mealeon_id_assigner("Cookstr", recipe_url)


def mealeon_id_epicurious_assigner(recipe_url: str) -> str:
    """Passthrough function to mealeon_id_assigner with 'Cookstr' as the origin. Doing it this way because polars map_elements doesn't seem to take *args or **kwargs"""
    return mealeon_id_assigner("Epicurious", recipe_url)


def allrecipes_id_extractor(recipe_url: str) -> str:
    """Takes in the AllRecipes URL and returns back the portion that seems to be a integer ID"""
    origin_id = urlparse(recipe_url).path.split("/")[-2]
    return origin_id
