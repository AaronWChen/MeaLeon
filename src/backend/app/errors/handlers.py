from flask import render_template
from src.backend.app import db
from src.backend.app.errors import bp


@bp.errorhandler(404)
def not_found_error(error):
    # display custom 404 error page
    return render_template("errors/404.html"), 404


@bp.errorhandler(500)
def internal_error(error):
    # display custom 500 error page and undo a database change to a clean state
    db.session.rollback()
    return render_template("errors/500.html"), 500
