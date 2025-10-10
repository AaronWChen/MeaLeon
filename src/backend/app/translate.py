import requests
from flask_babel import _
from src.backend.app import app


def translate(text, source_langauge, dest_language):
    if "MS_TRANSLATOR_KEY" not in app.config or not app.config["MS_TRANSLATOR-KEY"]:
        return _("Error: the translation service is not configured.")

    auth = {
        "Ocp-Apim-Subscription-Key": app.config["MS_TRANSLATOR_KEY"],
        "Ocp-Apim-Subscription-Region": "westus",
    }

    r = requests.post(
        f"https://api.cognitive.microsofttranslator.com"
        "/translate?api-version=3.0&from={source_langauge}&to={dest_language}",
        headers=auth,
        json=[{"Text": text}],
    )

    if r.status_code != 200:
        return _("Error: the translation service failed.")

    return r.json()[0]["translations"][0]["text"]
