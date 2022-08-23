import pandas as pd
import numpy as np
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from collections import OrderedDict
import pandas as pd
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import starmap_parallelized_eval
from multiprocessing.pool import ThreadPool
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.optimize import minimize
from pymoo.util.termination.default import MultiObjectiveSpaceToleranceTermination
from pymoo.core.evaluator import Evaluator
    
def randSample(mx, n):
    return np.random.choice(mx, n, replace = False)

class MyProblem(ElementwiseProblem):
    def __init__(self, df, n_sampled, **kwargs):
        super().__init__(n_var=n_sampled, n_obj=4, n_constr=2, xl=0, xu=len(df)-1, **kwargs)
        self.df = df
        genr = df.iterrows()
        self.charlist = []
        self.wordlist = []
        for i,row in df.iterrows():
            words = row['text'].split()
            words = list(OrderedDict.fromkeys(words).keys())
            self.charlist.append(words)
            self.wordlist.append(list(OrderedDict.fromkeys("".join(words)).keys()))
    def getStats(self, X):
        tot_chars = set()
        tot_words = set()
        tot_auths = {}
        tot_sents = {'+':0,'-':0,'0':0}
        for i in X:
            row = self.df.iloc[i]
            tot_chars.update(self.charlist[i])
            tot_words.update(self.wordlist[i])
            tot_auths[row['author']] = 1
            if isinstance(row['pysent_sent'], str):
                if row['pysent_sent'].startswith("POS"):
                    tot_sents['+']+=1
                    continue
                if row['pysent_sent'].startswith("NEG"):
                    tot_sents['-']+=1
                    continue
            tot_sents['0']+=1
        return tot_chars,tot_words,tot_auths,tot_sents
    def _evaluate(self, X, out, *args, **kwargs):
        tot_chars,tot_words,tot_auths,tot_sents = self.getStats(X)
        sents = abs(tot_sents['+'] - tot_sents['-'])+abs(tot_sents['+'] - tot_sents['0'])+abs(tot_sents['-'] - tot_sents['0'])
        out["F"] = np.column_stack([-len(tot_chars),-len(tot_words),-len(tot_auths), sents])
        out["G"] = np.column_stack([117-len(tot_chars), 49000-len(tot_words)])


class MySampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        pops = np.load('pops.npy', allow_pickle=True)
        return pops[-n_samples:]

class MyCrossover(Crossover):
    def __init__(self):
        super().__init__(2, 1)
    def _do(self, problem, X, **kwargs):
        n_parents, n_matings, n_var = X.shape
        _X = np.random.randint(0, len(problem.df)-1, (self.n_offsprings, n_matings, problem.n_var))
        for k in range(n_matings):
            p1, p2 = X[0, k], X[1, k]
            fh = np.intersect1d(p1, p2, assume_unique=True)
            n_remaining = p1.shape[0] - fh.shape[0]
            s1 = np.setdiff1d(p1, p2, assume_unique=True)
            s2 = np.setdiff1d(p2, p1, assume_unique=True)
            I  = np.concatenate((s1,s2))
            S = np.random.permutation(I)[:n_remaining]
            _X[0, k] = np.concatenate((fh, S))
        return _X

class MyMutation(Mutation):
    def _do(self, problem, X, **kwargs):
        for i in range(X.shape[0]):
            X[i, :] = X[i, :]
            tot_ch,tot_w, _, _ = problem.getStats(X[i, :])
            for t in randSample(len(problem.df), 2*X.shape[1]):
                if t not in X[i, :]:
                    chs, wrds, _, _ = problem.getStats([t])
                    flag = 0
                    for c in chs:
                        if c not in tot_ch:
                            flag+=50
                            break
                    for w in wrds:
                        if w not in tot_w:
                            flag+=1
                            break
                    if flag >= 50:
                        idx = np.random.choice(X[i, :].shape[0])
                        X[i,idx] = t
                        break
        return X

tdf=pd.read_csv("yml_nodups_labelled8.csv", sep='\t', encoding='utf-8')

n_threads = 8
pool = ThreadPool(n_threads)

problem = MyProblem(tdf, 5000, runner=pool.starmap, func_eval=starmap_parallelized_eval)

algorithm = NSGA2(
    pop_size=100,
    sampling=MySampling(),
    crossover=MyCrossover(),
    mutation=MyMutation(),
    eliminate_duplicates=True)

termination = MultiObjectiveSpaceToleranceTermination(n_max_evals=1000)

res = minimize(problem,
               algorithm,
               termination,
               seed=90210,
               verbose=True,
               save_history=True)

'''
print("Function value: %s" % res.F[0])
indices = np.where(res.X)[0]
print("Subset:", list(indices))
pool.close()
sdf=tdf.iloc[indices]
sdf.to_csv("yml_small.csv", sep='\t', encoding='utf-8', index=False)

class MySampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        X = randSample(len(df), n_samples*problem.n_var)
        return np.reshape(X,(-1, problem.n_var))
'''