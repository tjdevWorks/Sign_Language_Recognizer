import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value
print 
        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores 
        lowest_bic_score = np.inf
        best_model = None
        for num_state in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(num_state)
                log_likelihood = model.score(self.X, self.lengths)
                p = (num_state ** 2) + 2*num_state*len(self.lengths) - 1
                bic_score = -2 * log_likelihood + p * np.log(len(self.lengths))
                if lowest_bic_score > bic_score:
                    lowest_bic_score = bic_score
                    best_model = model
            except Exception as e:
                continue
        return best_model
        raise NotImplementedError


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        highest_dic_score = -np.inf
        best_model = None
        
        for num_state in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(num_state)
                log_likelihood = model.score(self.X, self.lengths)
                total_sum = 0
                for w in self.words:
                    if w!=self.this_word:
                        try:
                            w_x, w_length = self.hwords[w]
                            w_logL_score = model.score(w_x, w_length)
                        except:
                            w_logL_score = 0
                        total_sum += w_logL_score
                anti_likelihood = total_sum / (len(self.words)-1)
                dic_score = log_likelihood - anti_likelihood 
                if highest_dic_score < dic_score:
                    highest_dic_score = dic_score
                    best_model = model
            except Exception as e:
                continue
        return best_model
        raise NotImplementedError
    
class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        highest_cv_score = -np.inf
        best_model = None
        split_data = KFold()
        for num_states in range(self.min_n_components,self.max_n_components+1):
            try:
                if split_data.n_splits > len(self.lengths):
                    model = self.base_model(num_states)
                    cv_score = model.score(self.X, self.lengths)
                else:
                    score = 0
                    for cv_train_idx, cv_test_idx in split_data.split(self.sequences):
                        cv_train, train_lengths = combine_sequences(cv_train_idx, self.sequences)
                        cv_test, test_lengths = combine_sequences(cv_test_idx, self.sequences)
                        model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(cv_train,train_lengths)
                        score += model.score(cv_test, test_lengths)
                    cv_score = score / split_data.n_splits
                if highest_cv_score < cv_score:
                    highest_cv_score = cv_score
                    best_model = model
            except Exception as e:
                #print(str(e))
                continue
        return best_model
                
        raise NotImplementedError
