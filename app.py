from flask import Flask, send_from_directory, render_template, request, abort
from waitress import serve
from src.nltk.ohe_dish_predictor import find_similar_dishes
import json

app = Flask(__name__, static_url_path="/static")


@app.route("/")
def index():
    """Return the main page."""
    return send_from_directory("static", "index.html")


@app.route("/get_results", methods=["POST"])
def get_results():
    """Display the five most similar recipes from the database based on the
    inputs."""
    data = request.form
    print(data)

    expected_features = ("dish_name", "cuisine_name")

    if data and all(feature in data for feature in expected_features):
        # Convert the dict of fields into a list
        dish = data["dish_name"]
        cuisine = data["cuisine_name"]
        results, ingreds, rec_weights = find_similar_dishes(dish, cuisine)
        return render_template(
            "results.html",
            results=results,
            dish=dish,
            cuisine=cuisine,
            ingreds=ingreds,
            recipe_weights=rec_weights,
        )

    else:
        return abort(400)


if __name__ == "__main__":
    website_url = "mealeon.buenosds.dev:5000"
    app.config["SERVER_NAME"] = website_url
    serve(app, host="0.0.0.0", port=5000)
