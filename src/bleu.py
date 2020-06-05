import unittest
import numpy as np
import nltk
import math
from collections import Counter

def bleu_score(reference, hypothesis, weights=[1/4]*4):
  if len(hypothesis) == 0:
      return 0
  N = min(4,len(hypothesis))
  ref_lens = (len(ref) for ref in reference)
  closest = min(
      ref_lens, key=lambda ref_len: (abs(ref_len - len(hypothesis)), ref_len)
  )
  bp = min(1, math.exp(1-closest/len(hypothesis)))

  sm = 0
  for i in range(1,N+1):
    lh = []
    for k in range(len(hypothesis)-i+1):
      lh.append(' '.join(hypothesis[k:k+i]))
    ch = Counter(lh)
    s = 0
    for t in ch:
      mn = 0
      max_count = 0
      for ref in reference:
        lr = []
        for k in range(len(ref)-i+1):
          lr.append(' '.join(ref[k:k+i]))
        cr = Counter(lr)
        max_count = max(max_count, cr[t])
      mn = min(ch[t], max_count)
      s += mn / len(lh)
    if s==0:
      if i==1:
        return 0
      else:
        pass
    else:
      sm += weights[i-1] * math.log(s)
  return bp*math.exp(sm)


class TestBleu(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_hand_craft_case(self):
        
        nltk_score = nltk.translate.bleu_score.sentence_bleu(reference, hypothesis)
        my_score = bleu_score(reference, hypothesis)
        print (nltk_score, my_score)
        self.assertAlmostEqual(nltk_score, my_score)        

    def test_random_case(self):
        for i in range(1000):
            vocab  = np.array(['a','b','c','d','e'])
            n_word = np.random.randint(10)
            n_ref = np.random.randint(1,5)
            idx = np.random.choice(len(vocab), n_word)
            hypothesis = vocab[idx].tolist()
            reference = []
            for j in range(n_ref):
                n_word = np.random.randint(10)
                idx = np.random.choice(len(vocab), n_word)
                ref = vocab[idx].tolist()
                reference.append(ref)

            nltk_score = nltk.translate.bleu_score.sentence_bleu(reference, hypothesis)
            my_score = bleu_score(reference, hypothesis)
            self.assertAlmostEqual(nltk_score, my_score)

if __name__ == '__main__':
    unittest.main()