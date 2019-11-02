import unittest
import numpy as np
from hmmlearn.hmm import MultinomialHMM

class TestHMM(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.n_state = 2
        self.n_output = 3
        self.step = 10

    def test_viterbi(self):
        model = MultinomialHMM(n_components=self.n_state)
        p = np.random.random(self.n_state)
        model.startprob_ = p/p.sum()
        p = np.random.random((self.n_state,self.n_state))
        model.transmat_ = p/p.sum(axis=1).reshape(-1,1)
        p = np.random.random((self.n_state,self.n_output))
        model.emissionprob_ = p/p.sum(axis=1).reshape(-1,1)
        X = np.random.choice(self.n_output,self.step).reshape(-1,1)
        self.assertTrue(np.array_equal(model.predict(X),model.predict(X)))

class HMM:
    def __init__(self):
        pass
    def viterbi(self, startprob, transmat, emissionprob, X):
        print (X)
        


if __name__ == '__main__':
    #unittest.main()
    np.random.seed(0)
    n_state = 2
    n_output = 3
    step = 10
    p = np.random.random(n_state)
    startprob = p/p.sum()
    p = np.random.random((n_state,n_state))
    transmat = p/p.sum(axis=1).reshape(-1,1)
    p = np.random.random((n_state,n_output))
    emissionprob = p/p.sum(axis=1).reshape(-1,1)
    X = np.random.choice(n_output,step).reshape(-1,1)
    hmm = HMM()
    hmm.viterbi(startprob,transmat, emissionprob, X)