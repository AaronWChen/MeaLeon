import pickle

def predict_wine(test_value):
    loaded_model = load_model()
    predictions = loaded_model.predict([test_value])
    if len(predictions > 0):
        return predictions[0]
    else:
        return -1

def load_model():
    """ Load the model from the .pickle file """
    model_file = open("models/wine_classifier.pickle", "rb")
    loaded_model = pickle.load(model_file)
    model_file.close()
    return loaded_model