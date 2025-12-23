from flask import Blueprint
import click
import os

bp = Blueprint("cli", __name__, cli_group=None)


@bp.cli.group()
def translate():
    """Translation and localization commands."""
    pass


@translate.command()
def update():
    """Update all languages."""
    if os.system("pybabel extract -F babel.cfg -k _l -o messages.pot ."):
        raise RuntimeError("extract command failed")

    if os.system("pybabel update -i messages.pot -d src/backend/app/translations"):
        raise RuntimeError("update command failed")

    os.remove("messages.pot")


@translate.command()
def compile():
    """Compile all languages."""
    if os.system("pybabel compile -d src/backend/app/translations"):
        raise RuntimeError("compile command failed")


@translate.command()
@click.argument("lang")
def init(lang):
    """Initialize a new language."""
    if os.system("pybabel extract -F babel.cfg -k _l -o messages.pot ."):
        raise RuntimeError("extract command failed")

    if os.system(
        "pybabel init -i messages.pot -d src/backend/app/translations -l " + lang
    ):
        raise RuntimeError("init command failed")
    os.remove("messages.pot")
