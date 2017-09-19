import warnings
from asl_data import SinglesData
import numpy as np

def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    for i in range(0, test_set.num_items):
        test_x, test_lengths = test_set.get_item_Xlengths(i)
        best_guess_score = -np.inf
        best_guess_word = None
        prob_words = dict()
        for word, model in models.items():
            try:
                score = model.score(test_x, test_lengths)
            except:
                score = -np.inf
            if best_guess_score < score:
                best_guess_score = score
                best_guess_word = word
            prob_words[word] = score
        probabilities.append(prob_words)
        guesses.append(best_guess_word)

    return probabilities, guesses