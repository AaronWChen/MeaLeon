- [What is MeaLeon?](#what-is-mealeon)
- [How did MeaLeon come about?](#how-did-mealeon-come-about)
- [How it works](#how-it-works)
- [Future Steps](#future-steps)
  - [To Do](#to-do)
- [Requirements](#requirements)
- [Running the Flask Application](#running-the-flask-application)
  - [Running from CLI](#running-from-cli)
  - [Running from Docker Container](#running-from-docker-container)
  - [Feature Requests](#feature-requests)
 
# What is MeaLeon?
MeaLeon is a machine learning powered dish and recipe recommendation tool. It 
asks a user to type a dish that they like and choose the cuisine type it is
from. It then presents the five most similar recipes back to the user with 
links to their location on Epicurious.

# How did MeaLeon come about?
There are a few origin stories for MeaLeon.
1. One of the creator's friends hates sauerkraut and kimchi, and expressed 
their disgust by saying "every culture has a gross pickled cabbage dish."

2. One day, the creator was making ground beef tacos but ran out of cumin. In
desperation, he grabbed something that he thought contained mostly cumin and 
used that: garam masala powder. Taco Tuesday suddenly had an Indian twist 
despite keeping most of the ingredients the same. 

3. The creator regularly cooks, but tends to make the same dishes over and over
and wanted something/someone to help him make new dishes without buying new 
spices and herbs for single usage.

All of those combined to form a desire to make something that would help him
find relationships between different cuisines and make new foods!

# How it works
The database was created from a 2017 scrape of Epicurious.com. The recipes in 
this database have been cleaned and converted to a matrix that is used via
natural language processing and machine learning. The ingredients for each 
recipe have been transformed into vectors via term frequency, inverse document
frequency techniques using the toolkit provided by scikit-learn.

The app uses the user inputs to call upon the Edamam API and takes the top 10
recipes returned to create a query vector that is then compared to the database
recipes via cosine similarity.

This particular branch (`tfidf_master`) is intended to be the standard branch that is used for MeaLeon.

# Future Steps
The database, from a foodie's point of view, could use more data.
1. The cuisine classifications do not dive into deep classifications for many
cuisine types and this should be fleshed out.

2. In an effort to add more recipes to the database, other sites and recipes
will be scraped, classified, and added to the database. If you know of any 
recipes or recipe sites, especially for the cuisines of Africa, Southeast Asia,
Oceania, and South America, please help!

3. Other recipes should ideally contribute to under represented cuisines in the
database.

For 2 and 3:
Finding new recipes and adding to the database will be moved into a new repo. The libraries and scripts needed for that work is different and lighter weight. Moving the scraping to a new repo should allow for that work to be done on less powerful computing hardware and represents a new MVP.

4. A classification algorithm will need to be used to estimate the cuisine 
classifications of new recipes.

5. That algorithm will then need to incorporate the ability to take multi-label classifications: For example, an ideal classification for Dandan noodles 
should return Sichuan, Chinese, and East Asian for cuisine.

6. Other implementations of this would be interesting for home cooks. One 
example would be an Alexa skill or Google Home integration to display or read 
aloud the proposed recipe steps and ingredients via smart home devices.

7. After discussion with [Karen Warmbein](https://github.com/karenkathryn), I will attempt a model using Word2Vec with a CBoW architecture using a CBoW window of 1 word to see the difference. She suggested that the initial training would speed up as compared to TF-IDF and I wonder if the size of the deployed app would be smaller.

## To Do
1. Incorporate a better template for notebook files
2. Refactor to use [spaCy](https://spacy.io/) or [stanza](https://github.com/stanfordnlp/stanza) over NLTK
3. Update to new git branch architecture: main with parallel dev branch, branch has sub parallels with features. Each feature branch starts with date_initials_feature
4. Update heroku deployment as it is currently down (likely due to security vulnerability)
5. Incorporate Data Version Control for database and relational data store for MLFlow
6. Source for SQL database for MLFlow and also object store for Data Version Control
7. Docker container alternate installation instructions/set up
8. Update database of recipes
9. Switch off of Edamam API or update knowledge of how it works since the results it returns have changed significantly.

# Requirements
This repo has switched to using Python 3.10 via ASDF and Python package management via Poetry.

If you've got both installed (INSTRUCTIONS PENDING), you can go into your Terminal/CLI, go to the base of this repository, and type 
```bash 
poetry install
``` 
to get the base installation working.



# Running the Flask Application
## Running from CLI
To run in a development environment (on your local computer)

(CHECK THESE INSTRUCTIONS)

```bash
export FLASK_ENV=development
env FLASK_APP=app.py flask run
```

To run in a production environment (used for deployment, but test it out 
locally first):
```bash
export FLASK_ENV=production
python app.py
```

## Running from Docker Container

If running via Docker, the CLI instructions I used that worked were
```bash
docker run -it --rm -p 5000:5000 awashingtonchen/mealeon_docker
```

The -p flag is needed to specify the port. The --rm flag is used to clean up the container afterwards (good for quick tests on development). The -it tag is for interactive and I used it initially because there are some bash commands inside the script to download NLTK if it isn't installed, or to access files inside the container.

## Feature Requests

1. Quarto Notebook
2. Refactor from Pandas to polaRS?
