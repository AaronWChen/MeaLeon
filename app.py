import sqlalchemy as sa
import sqlalchemy.orm as so
from src.backend.app import create_app, db, cli
from src.backend.app.models import User, Review  # , Allergy


app = create_app()


@app.shell_context_processor
def make_shell_context():
    return {
        "sa": sa,
        "so": so,
        "db": db,
        "User": User,
        "Review": Review,
        # "Allergy": Allergy,
    }
