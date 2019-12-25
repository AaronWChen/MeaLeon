# Why joblib files

This folder contains byte data containing the overall recipe database
(recipe_dataframe.joblib), the recipe TFIDF object that is used to transform 
new data (recipe_tfidf.joblib), the TFIDF word matrix formed from the database
and transformer (recipe_word_matrix.joblib), and the portion of the original
dataset that has been left out of the training set.

The joblib files are used for speed and reproducibility. Loading, cleaning, 
and prepping the original dataset adds unnecessary time to running the model. 
Likewise, creating and training the TFIDF transformer is an unnecessary step
for the user.

# Future Steps

The database and model should be occasionally updated if new recipes are found
and added.

To save time and avoid unnecessary training, this should occur no more 
frequently than quarterly.