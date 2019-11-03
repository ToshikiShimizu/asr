import unittest
import numpy as np
from hmmlearn.hmm import MultinomialHMM

class TestHMM(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_viterbi_case_handcraft(self):
        # init
        startprob = np.array([0.6, 0.4])
        transmat = np.array([[0.7, 0.3],
                                    [0.4, 0.6]])
        emissionprob = np.array([[0.1, 0.4, 0.5],
                                    [0.6, 0.3, 0.1]])
        X = np.array([1,0,2,0,2,1,0,1,1]).reshape(-1,1)

        # hmmlearn
        model = MultinomialHMM(n_components=2)
        model.startprob_ = startprob
        model.transmat_ = transmat
        model.emissionprob_ = emissionprob
        y = model.predict(X)

        # my hmm
        hmm = HMM()
        pred = hmm.viterbi(startprob, transmat, emissionprob, X)
        self.assertTrue(np.array_equal(y, pred))

    def test_viterbi_case_random(self):
        for i in range(1000):
            # init
            self.n_state = np.random.randint(1,10)
            self.n_output = np.random.randint(1,10)
            self.step = np.random.randint(1,200)
            p = np.random.random(self.n_state)
            startprob = p/p.sum()
            p = np.random.random((self.n_state,self.n_state))
            transmat = p/p.sum(axis=1).reshape(-1,1)
            p = np.random.random((self.n_state,self.n_output))
            emissionprob = p/p.sum(axis=1).reshape(-1,1)
            X = np.random.choice(self.n_output,self.step).reshape(-1,1)

            # hmmlearn
            model = MultinomialHMM(n_components=self.n_state,)
            model.startprob_ = startprob
            model.transmat_ = transmat
            model.emissionprob_ = emissionprob
            y = model.predict(X)

            # my hmm
            hmm = HMM()
            pred = hmm.viterbi(startprob, transmat, emissionprob, X)
            self.assertTrue(np.array_equal(y, pred))

class HMM: 
    def __init__(self):
        pass

    def viterbi(self, prob, transmat, emissionprob, X):
        prob = np.log(prob) #  アンダーフローを防ぐために対数をとる。hmmlearnの実装もそう
        transmat = np.log(transmat)
        emissionprob = np.log(emissionprob)
        history = []
        for i in range(len(X)):
            prob = prob + emissionprob[:,X[i]].ravel()
            if (i==len(X)-1): 
                break # 出力はlen(X)回だが、遷移はlen(X)-1回
            history.append(np.argmax(transmat.T + prob,axis=1)) # パスを保存して遷移
            prob = np.max(transmat.T + prob,axis=1)            
        s = np.argmax(prob) # 終了状態の累積確率からバックトラック開始
        seq = [s]
        for h in history[::-1]:
            s = h[s]
            seq.append(s)
        return seq[::-1]

if __name__ == '__main__':
    unittest.main()