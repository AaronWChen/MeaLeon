from app import db

from app.main import bp
from app.main.forms import (
    EditProfileForm,
    EmptyForm,
    ReviewForm,
)
from app.models import User, Review
from app.translate import translate
from .routes_get_similar_dishes import get_similar_dishes

# from src.nltk import dish_predictor as dp  # import find_similar_dishes

import httpx

from datetime import datetime, timezone
from flask import (
    render_template,
    request,
    abort,
    flash,
    redirect,
    url_for,
    g,
    current_app,
)
from flask_babel import _, get_locale
from flask_login import current_user, login_required
from langdetect import detect, LangDetectException
import sqlalchemy as sa
import sqlalchemy.orm as so


@bp.before_request
def before_request():
    if current_user.is_authenticated:
        current_user.last_seen = datetime.now(timezone.utc)
        db.session.commit()
    g.locale = str(get_locale())


@bp.route("/")
@bp.route("/index")
def index():
    return render_template("index.html")


@bp.route("/user/<username>")
@login_required
def user_profile(username):
    # want this to display their most recently "to cook" recipes and allow updating of account info
    # maybe show "how many to cook" and "recipes cooked"
    user = db.first_or_404(sa.select(User).where(User.username == username))

    # reviews = [
    #     {"author": user, "body": "Review text 1"},
    #     {"author": user, "body": "Review text 2"},
    # ]

    form = EmptyForm()

    return render_template("user_profile.html", user=user, form=form)


@bp.route("/edit_profile", methods=["GET", "POST"])
@login_required
def edit_profile():
    # allow users to update their profile
    form = EditProfileForm(current_user.username)

    if form.validate_on_submit():
        current_user.username = form.username.data
        current_user.about_me = form.about_me.data
        db.session.commit()
        flash(_("Your changes have been saved!"))
        return redirect(url_for("main.edit_profile"))

    elif request.method == "GET":
        form.username.data = current_user.username
        form.about_me.data = current_user.about_me

    return render_template("edit_profile.html", title=_("Edit Profile"), form=form)


@bp.route("/follow/<username>", methods=["POST"])
@login_required
def follow(username):
    form = EmptyForm()
    if form.validate_on_submit():
        user = db.session.scalar(sa.select(User).where(User.username == username))

        if user is None:
            flash(_("User %(username) not found.", username=username))
            return redirect(url_for("main.index"))

        if user == current_user:
            flash(_("You cannot follow yourself"))
            return redirect(url_for("main.user_profile", username=username))

        current_user.follow(user)
        db.session.commit()
        flash(_("You are now following %(username)!", username=username))
        return redirect(url_for("main.user_profile", username=username))
    else:
        return redirect(url_for("main.index"))


@bp.route("/unfollow/<username>", methods=["POST"])
@login_required
def unfollow(username):
    form = EmptyForm()
    if form.validate_on_submit():
        user = db.session.scalar(sa.select(User).where(User.username == username))

        if user is None:
            flash(_("User %(username) not found.", username=username))
            return redirect(url_for("main.index"))

        if user == current_user:
            flash(_("You cannot unfollow yourself!"))
            return redirect(url_for("main.user_profile", username=username))

        current_user.unfollow(user)
        db.session.commit()
        flash(_("You are no longer following %(username)", username=username))
        return redirect(url_for("main.user_profile", username=username))
    else:
        return redirect(url_for("main.index"))


@bp.route("/write_review", methods=["GET", "POST"])
@login_required
def write_review():
    form = ReviewForm()

    # display reviews from current user
    if form.validate_on_submit():
        try:
            language = detect(form.review.data)
        except LangDetectException:
            language = ""

        review = Review(body=form.review.data, author=current_user, language=language)
        db.session.add(review)
        db.session.commit()
        flash(_("Your review is now live!"))
        return redirect(url_for("main.user_profile", username=current_user.username))

    reviews = db.session.scalars(current_user.personal_reviews()).all()

    return render_template(
        "index.html",
        title=_("Home Page"),
        # form=form,
        # reviews=reviews,
    )


@bp.route("/user/<username>/reviews", methods=["GET", "POST"])
@login_required
def user_reviews(username):
    user = db.first_or_404(sa.select(User).where(User.username == username))

    Author = so.aliased(User)
    page = request.args.get("page", 1, type=int)
    query = (
        sa.select(Review).where(Author.id == user.id).order_by(Review.timestamp.desc())
    )
    reviews = db.paginate(
        query,
        page=page,
        per_page=current_app.config["REVIEWS_PER_PAGE"],
        error_out=True,
    )

    next_url = (
        url_for("main.user_profile", page=reviews.next_num)
        if reviews.has_next
        else None
    )

    prev_url = (
        url_for("main.user_profile", page=reviews.prev_num)
        if reviews.has_prev
        else None
    )

    return render_template(
        "user_profile.html",
        title=_("Reviews"),
        user=user,
        reviews=reviews.items,
        next_url=next_url,
        prev_url=prev_url,
    )


@bp.route("/explore")
@login_required
def explore():
    page = request.args.get("page", 1, type=int)
    query = sa.select(Review).order_by(Review.timestamp.desc())
    reviews = db.paginate(
        query,
        page=page,
        per_page=current_app.config["REVIEWS_PER_PAGE"],
        error_out=True,
    )

    next_url = (
        url_for("main.user_profile", page=reviews.next_num)
        if reviews.has_next
        else None
    )

    prev_url = (
        url_for("main.user_profile", page=reviews.prev_num)
        if reviews.has_prev
        else None
    )

    return render_template(
        "write_review.html",
        user=current_user,
        title=_("Explore"),
        reviews=reviews.items,
        next_url=next_url,
        prev_url=prev_url,
    )


@bp.route("/translate", methods=["POST"])
@login_required
def translate_text():
    data = request.get_json()
    return {
        "text": translate(data["text"], data["source_language"], data["dest_language"])
    }


# Replace the route body:
def get_similar_dishes(dish, cuisine):
    """
    Replaces dp.find_similar_dishes() — calls search_service instead.
    Returns (results, ingreds, rec_weights) to match the original signature
    so results.html doesn't need to change.
    """
    try:
        resp = httpx.post(
            current_app.config["SEARCH_SERVICE_URL"] + "/search",
            json={"dish_name": dish, "cuisine": cuisine, "max_results": 5},
            timeout=15.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        current_app.logger.error(f"Search service error: {e}")
        return [], [], []

    recipes = data.get("recipes", [])

    # Reshape Edamam results to match the shape results.html expects
    # from the old Epicurious data — map field names across
    results = [
        {
            "hed": r.get("label", ""),
            "title": r.get("label", ""),  # same as hed
            "fixed_url": r.get("url", ""),  # was url
            "photo": r.get("image_url") or "",  # was filename
            "imputed_label": ", ".join(r.get("cuisine_types", [])),
            "ingred_weights": r.get("ingredient_names", [])[:5],
            "rounded": round(1.0, 4),  # placeholder score
            "ingredients": r.get("ingredient_lines", []),
            "source": r.get("source", ""),
        }
        for r in recipes
    ]

    # ingreds — flat deduplicated list across all recipes
    seen = set()
    ingreds = [
        i
        for r in recipes
        for i in r.get("ingredient_names", [])
        if not (i in seen or seen.add(i))
    ]

    # rec_weights — similarity scores (Edamam doesn't provide these,
    # use placeholder 1.0 until recommend_service is wired in)
    rec_weights = [1.0] * len(results)

    return results, ingreds, rec_weights


@bp.route("/get_results", methods=["GET", "POST"])
def get_results():
    data = request.form
    expected_features = ("dish_name", "cuisine_name")
    if data and all(feature in data for feature in expected_features):
        dish = data["dish_name"]
        cuisine = data["cuisine_name"]
        results, ingreds, rec_weights = get_similar_dishes(dish, cuisine)
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

    # reviews = [
    #     {
    #         'author': {'username':'John'},
    #         'review': 'Really good pasta'
    #     },
    #     {
    #         'author': {'username': 'Sarah'},
    #         'review': 'Liked the custard'
    #     }
    # ]

    # after this is a predetermined test recipe and results, in case we need to display results page without working nlp pipeline
    # dish = "Lasagna"
    # cuisine = "Italian"
    # results = [
    #     {
    #         "hed": "Lean Lasagna",
    #         "filename": "lean-lasagna.jpg",
    #         "imputed_cuisine_name": "American",
    #         "ingredients": [
    #             "Vegetable-oil cooking spray",
    #             "1/2 cup chopped onion",
    #             "1 lb ground turkey breast",
    #             "3 cups tomato sauce",
    #             "3 tsp Italian seasoning (or 1 tsp each dried basil, parsley, and oregano)",
    #             "1/4 tsp freshly ground black pepper",
    #             "1/4 tsp garlic powder",
    #             "1/2 cup chopped mushrooms",
    #             "6 cups chopped fresh spinach (or chard)",
    #             "2 cups fat-free ricotta",
    #             "1/4 tsp nutmeg",
    #             "1 package whole-wheat lasagna noodles(about 8 oz, or 9 noodles)",
    #             "2 cups (8 oz) shredded part-skim mozzarella",
    #         ],
    #         "cosine_similarity": 0.5825168147272504,
    #         "photo": "photos/lean-lasagna.jpg",
    #         "fixed_url": "https://www.epicurious.com/recipes/food/views/lean-lasagna-230145",
    #         "rounded": 0.583,
    #         "ingred_weights": {
    #             "Vegetable": 0.2202291486446679,
    #             "basil": 0.1696630151682376,
    #             "black": 0.11266830989083243,
    #             "chard": 0.26733217817901767,
    #             "fat": 0.18271061927822305,
    #             "garlic": 0.09912628327558201,
    #             "lasagna": 0.3366539727958394,
    #             "mozzarella": 0.23242817372190097,
    #             "mushrooms": 0.1828039297807702,
    #             "noodles": 0.4756800809877015,
    #             "nutmeg": 0.19051981989954536,
    #             "oil": 0.07590798020679795,
    #             "onion": 0.1183317527555511,
    #             "oregano": 0.1888055809506423,
    #             "parsley": 0.13566717719268898,
    #             "pepper": 0.09726959119058161,
    #             "ricotta": 0.23620837812317497,
    #             "sauce": 0.14540446475446236,
    #             "spinach": 0.20741091003045053,
    #             "tomato": 0.18513870145898717,
    #             "turkey": 0.21436776114597628,
    #             "wheat": 0.22030304927926814,
    #         },
    #     },
    #     {
    #         "hed": "Butternut Squash and Mushroom Lasagna",
    #         "filename": "231091.jpg",
    #         "imputed_cuisine_name": "American",
    #         "ingredients": [
    #             "1/4 cup (1/2 stick) unsalted butter",
    #             "2 1/2 cups chopped onions",
    #             "1/2 pound crimini (baby bella) mushrooms, sliced (about 3 cups)",
    #             "2 pounds butternut squash, peeled, seeded, cut into 1/4-inch-thick slices (about 5 1/2 cups)",
    #             "1 14-ounce can vegetable broth",
    #             "4 tablespoons chopped fresh thyme, divided",
    #             "4 tablespoons sliced fresh sage, divided",
    #             "3 15-ounce containers whole-milk ricotta cheese",
    #             "4 cups grated mozzarella cheese, divided",
    #             "2 cups grated Parmesan cheese, divided",
    #             "4 large eggs",
    #             "Olive oil",
    #             "1 9-ounce package no-boil lasagna noodles",
    #         ],
    #         "cosine_similarity": 0.541084915756467,
    #         "photo": "photos/231091.jpg",
    #         "fixed_url": "https://www.epicurious.com/recipes/food/views/butternut-squash-and-mushroom-lasagna-231091",
    #         "rounded": 0.541,
    #         "ingred_weights": {
    #             "Olive": 0.22804478048438392,
    #             "Parmesan": 0.17169395738239085,
    #             "baby": 0.18084014458447717,
    #             "bella": 0.31766425634079537,
    #             "broth": 0.13781597649683697,
    #             "butter": 0.08912047577987173,
    #             "butternut": 0.23992363336485634,
    #             "cheese": 0.38941173430142334,
    #             "crimini": 0.26420592695834316,
    #             "eggs": 0.12637862705686756,
    #             "lasagna": 0.3261867787384241,
    #             "milk": 0.1310946427892246,
    #             "mozzarella": 0.22520155233807085,
    #             "mushrooms": 0.1771202178329126,
    #             "noodles": 0.23044515417245154,
    #             "oil": 0.07354786084526932,
    #             "onions": 0.14494345496616443,
    #             "ricotta": 0.2288642231997397,
    #             "sage": 0.2048636754392321,
    #             "squash": 0.21309662042398902,
    #             "thyme": 0.1502258765813285,
    #             "vegetable": 0.1243640237105734,
    #         },
    #     },
    #     {
    #         "hed": "Vegetable Lasagna",
    #         "filename": "vegetable-lasagna.jpg",
    #         "imputed_cuisine_name": "American",
    #         "ingredients": [
    #             "7 oz lowfat goat cheese",
    #             "1/3 cup chopped pitted black olives",
    #             "1 tbsp chopped fresh thyme",
    #             "1/2 tbsp dried basil",
    #             "1/2 tbsp dried oregano",
    #             "2 tsp minced garlic",
    #             "4 cups prepared pasta sauce",
    #             "1 lb whole-wheat lasagna",
    #             "Freshly ground black pepper",
    #             "2 small zucchini, diced",
    #             "2 small summer squash, diced",
    #             "3/4 cup bottled roasted red pepper, diced",
    #             "1/4 cup grated Parmesan",
    #         ],
    #         "cosine_similarity": 0.4405954094530839,
    #         "photo": "photos/vegetable-lasagna.jpg",
    #         "fixed_url": "https://www.epicurious.com/recipes/food/views/vegetable-lasagna-232660",
    #         "rounded": 0.441,
    #         "ingred_weights": {
    #             "Freshly": 0.21818516240369507,
    #             "Parmesan": 0.1925454596506645,
    #             "basil": 0.18435208854382587,
    #             "black": 0.24484579883814772,
    #             "bottled": 0.23399042206234946,
    #             "cheese": 0.14556804544852314,
    #             "garlic": 0.10770843211362154,
    #             "goat": 0.2463823832935765,
    #             "lasagna": 0.36580077832489233,
    #             "olives": 0.21617867218900785,
    #             "oregano": 0.20515197812833288,
    #             "pasta": 0.23529870713819395,
    #             "pepper": 0.21138198292664606,
    #             "red": 0.1206070180176903,
    #             "sauce": 0.15799328294679807,
    #             "squash": 0.23897630036075063,
    #             "summer": 0.33811882771876234,
    #             "thyme": 0.16847017157018776,
    #             "wheat": 0.23937643219964208,
    #             "zucchini": 0.2320249109532601,
    #         },
    #     },
    #     {
    #         "hed": "Oven-Dried Tomato Tart with Goat Cheese and Black Olives",
    #         "filename": "232540.jpg",
    #         "imputed_cuisine_name": "American",
    #         "ingredients": [
    #             "5 tablespoons extra-virgin olive oil, divided",
    #             "6 medium tomatoes or large romas, cored, halved crosswise, seeded",
    #             "2 small garlic cloves, thinly slivered",
    #             "2 tablespoons minced fresh thyme, divided",
    #             "1 sheet frozen puff pastry (half of 17.3-ounce package), thawed",
    #             "1 cup coarsely grated whole-milk mozzarella cheese",
    #             "1/2 cup soft fresh goat cheese (about 4 ounces)",
    #             "2 large eggs",
    #             "1/4 cup whipping cream",
    #             "1/3 cup oil-cured black olives, pitted",
    #             "2 tablespoons freshly grated Parmesan cheese",
    #         ],
    #         "cosine_similarity": 0.28034633137531173,
    #         "photo": "photos/232540.jpg",
    #         "fixed_url": "https://www.epicurious.com/recipes/food/views/oven-dried-tomato-tart-with-goat-cheese-and-black-olives-232540",
    #         "rounded": 0.28,
    #         "ingred_weights": {
    #             "Parmesan": 0.22908479615993133,
    #             "black": 0.1456550832702511,
    #             "cheese": 0.5195774454429124,
    #             "cream": 0.14711034451125432,
    #             "eggs": 0.1686222535707234,
    #             "garlic": 0.12814825267872595,
    #             "goat": 0.2931383485053902,
    #             "milk": 0.17491465616429802,
    #             "mozzarella": 0.30047796963155227,
    #             "oil": 0.1962642945227558,
    #             "olive": 0.12164467952577326,
    #             "olives": 0.2572028815553146,
    #             "pastry": 0.2844217037400915,
    #             "puff": 0.3290586572588967,
    #             "thyme": 0.20044074258206962,
    #             "tomatoes": 0.18496669863323473,
    #         },
    #     },
    #     {
    #         "hed": "Pizza Noodles",
    #         "filename": "EP_12162015_placeholders_rustic.jpg",
    #         "imputed_cuisine_name": "American",
    #         "ingredients": [
    #             "1 pound ziti, rigatoni, or shells",
    #             "1 tablespoon olive oil",
    #             "3 cups Bolognese meat sauce, marinara tomato sauce, or store-bought pasta sauce",
    #             "1 pound ricotta cheese",
    #             "1 pound mozzarella, coarsely shredded or thinly sliced",
    #             "Grated Pecorino Romano or Parmesan cheese, for serving",
    #         ],
    #         "cosine_similarity": 0.26654363143114707,
    #         "photo": "photos/EP_12162015_placeholders_rustic.jpg",
    #         "fixed_url": "https://www.epicurious.com/recipes/food/views/pizza-noodles-235223",
    #         "rounded": 0.267,
    #         "ingred_weights": {
    #             "Bolognese": 0.38150004046148356,
    #             "Grated": 0.23258180236588194,
    #             "Parmesan": 0.1505895940744791,
    #             "Pecorino": 0.2279336600036965,
    #             "Romano": 0.22166388204529933,
    #             "cheese": 0.2276972192860822,
    #             "marinara": 0.2764286925271907,
    #             "meat": 0.20516172862647306,
    #             "mozzarella": 0.19752011584194992,
    #             "oil": 0.06450746828013475,
    #             "olive": 0.07996350355052645,
    #             "pasta": 0.18402686232372076,
    #             "ricotta": 0.2007325767036838,
    #             "rigatoni": 0.2778733142034248,
    #             "sauce": 0.3706991230323285,
    #             "shells": 0.2279336600036965,
    #             "tomato": 0.15733298237227247,
    #             "ziti": 0.3019594736209478,
    #         },
    #     },
    # ]
    # ingreds = {
    #     "lasagna": 0.7371297923338641,
    #     "noodles": 0.2603845399505066,
    #     "cheese": 0.2566690143549585,
    #     "ricotta": 0.19394866106969197,
    #     "mozzarella": 0.19084476785463195,
    # }
    # rec_weights = {
    #     "acorn": 0.029753069353458013,
    #     "baby": 0.01702790483246407,
    #     "bacon": 0.01741224209976635,
    #     "baking": 0.013414599108161512,
    #     "basic": 0.03387124719881589,
    #     "basil": 0.12383007800667767,
    #     "bay": 0.016340204985654385,
    #     "beef": 0.05170494415424768,
    #     "black": 0.05139498725174357,
    #     "breadcrumbs": 0.020671381685224022,
    #     "butter": 0.041957912157460094,
    #     "carrot": 0.01899338407691437,
    #     "carrots": 0.016311616728637606,
    #     "celery": 0.01542666072962034,
    #     "cheese": 0.2566690143549585,
    #     "cherry": 0.019144189330607093,
    #     "chicken": 0.011951053108193459,
    #     "chive": 0.0334546244833556,
    #     "chives": 0.016978553071633606,
    #     "chunky": 0.028917498571980508,
    #     "close": 0.0409563273245225,
    #     "clove": 0.07375413429698983,
    #     "coconut": 0.036821787929688654,
    #     "cover": 0.028334938565196272,
    #     "cream": 0.020763392834638517,
    #     "cremini": 0.052608366449550116,
    #     "curry": 0.02020823774485093,
    #     "dairy": 0.035158051494468615,
    #     "egg": 0.012003840798570274,
    #     "eggplants": 0.024946000483657423,
    #     "eggs": 0.011899809300232824,
    #     "flour": 0.03073269614853128,
    #     "fontina": 0.03707787490578407,
    #     "fresco": 0.027686883619768683,
    #     "garlic": 0.0633046834400603,
    #     "get": 0.03760589255117122,
    #     "goat": 0.020687011185841358,
    #     "green": 0.012240790332827113,
    #     "lasagna": 0.7371297923338641,
    #     "leaf": 0.015172105752621,
    #     "lemon": 0.01023141253786846,
    #     "love": 0.03981875804189533,
    #     "marinara": 0.14838142611123045,
    #     "mascarpone": 0.024700869371290967,
    #     "meat": 0.04405069473610725,
    #     "milk": 0.08640708481034252,
    #     "mozzarella": 0.19084476785463195,
    #     "mushrooms": 0.03335527318996653,
    #     "noodles": 0.2603845399505066,
    #     "nutmeg": 0.05214472671736605,
    #     "oil": 0.07617791811549247,
    #     "olive": 0.08584563769559089,
    #     "olives": 0.018151016115407998,
    #     "onion": 0.043182747164598204,
    #     "pancetta": 0.02384886955134506,
    #     "parmesan": 0.18908530352142724,
    #     "parmigiano": 0.11945627412568599,
    #     "parsley": 0.01237723872676221,
    #     "paste": 0.05042274033408072,
    #     "peas": 0.037331017767898274,
    #     "pecorino": 0.028281138299796133,
    #     "pepper": 0.11536376511347184,
    #     "pepperoni": 0.03307774339395352,
    #     "portobello": 0.05357233414621853,
    #     "potatoes": 0.015017912961645601,
    #     "pumpkin": 0.021240840742290417,
    #     "red": 0.04050612219056292,
    #     "reggiano": 0.07963751608379066,
    #     "ricotta": 0.19394866106969197,
    #     "rosemary": 0.016866732426197024,
    #     "sage": 0.03857992015017982,
    #     "salt": 0.06167784545814565,
    #     "sauce": 0.15918713511495225,
    #     "sausage": 0.06350829021012737,
    #     "shallots": 0.01680758011136024,
    #     "soy": 0.016524405575359925,
    #     "spinach": 0.056767769502619865,
    #     "squash": 0.04013034806001668,
    #     "stock": 0.018349721895056865,
    #     "sugar": 0.008019660138920347,
    #     "thyme": 0.028290531791799165,
    #     "tomato": 0.06756257343708133,
    #     "tomatoes": 0.06526625027300913,
    #     "veal": 0.02259120509653947,
    #     "vegetable": 0.011710114284589679,
    #     "vinegar": 0.01140737841031653,
    #     "water": 0.05024362635771761,
    #     "white": 0.020827232388144927,
    #     "wine": 0.0466667125439539,
    #     "yellow": 0.031981852492164405,
    #     "zucchini": 0.05844454295944473,
    # }
    # return render_template(
    #     "results.html",
    #     results=results,
    #     dish=dish,
    #     cuisine=cuisine,
    #     ingreds=ingreds,
    #     recipe_weights=rec_weights,
    # )
