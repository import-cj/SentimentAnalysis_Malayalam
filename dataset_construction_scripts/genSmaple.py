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

np.random.seed(73)    
def randSample(mx, n):
    return np.random.choice(mx, n, replace = False)

class MyProblem(ElementwiseProblem):
    def __init__(self, df, n_sampled, **kwargs):
        super().__init__(n_var=len(df), n_obj=4, n_constr=1, xl=0,xu=1)
        self.df = df
        self.n_sampled = n_sampled
        self.charlist = []
        self.wordlist = []
        for i,row in df.iterrows():
            words = set(row['text'].split())
            self.wordlist.append(words)
            chars = set()
            for w in words:
                chars.update(w)
            self.charlist.append(chars)
    def _evaluate(self, X, out, *args, **kwargs):
        tot_chars = set()
        tot_words = set()
        tot_auths = {}
        tot_sents = {'+':0,'-':0,'0':0}
        tot = 0
        for i in range(len(X)):
            if X[i] == False:
                continue
            tot+=1
            row = self.df.iloc[i]
            tot_words.union(self.wordlist[i])
            tot_chars.union(self.charlist[i])
            tot_auths[row['author']] = 1
            if isinstance(row['pysent_sent'], str):
                if row['pysent_sent'].startswith("POS"):
                    tot_sents['+']+=1
                    continue
                if row['pysent_sent'].startswith("NEG"):
                    tot_sents['-']+=1
                    continue
            tot_sents['0']+=1
        sents = abs(tot_sents['+'] - tot_sents['-'])+ abs(tot_sents['+'] - tot_sents['0'])+ abs(tot_sents['-'] - tot_sents['0'])
        out["F"] = np.column_stack([-len(tot_chars),-len(tot_words),-len(tot_auths), sents])
        out["G"] = tot - self.n_sampled - 100
'''
        return -len(tot_chars),-len(tot_words),-len(tot_auths), sents,tot

    def _evaluate(self, X, out, *args, **kwargs):
        f = [ [], [] , [] , []]
        g = []
        for x in X:
            fs = self._evalone(x):
            for i in range(4):
                f[i].append(fs[i])
            g.append(fs[5] - self.n_sampled - 100)
        out["F"] = np.column_stack(f)
        out["G"] = g
'''

class MySampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, problem.n_var), False, dtype=bool)
        for k in range(n_samples):
            I = np.random.permutation(problem.n_var)[:problem.n_sampled]
            X[k, I] = True
        return X

class MySampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, problem.n_var), False, dtype=bool)
        pops = np.load('pops.npy', allow_pickle=True)
        pops = pops[:-n_samples+1]
        for k in range(n_samples):
            X[k, pops[-k]] = True
        return X

class MyCrossover(Crossover):
    def __init__(self):
        super().__init__(2, 1)
    def _do(self, problem, X, **kwargs):
        n_parents, n_matings, n_var = X.shape
        _X = np.full((self.n_offsprings, n_matings, problem.n_var), False)
        for k in range(n_matings):
            p1, p2 = X[0, k], X[1, k]
            both_are_true = np.logical_and(p1, p2)
            _X[0, k, both_are_true] = True
            n_remaining = problem.n_sampled - np.sum(both_are_true)
            I = np.where(np.logical_xor(p1, p2))[0]
            S = I[np.random.permutation(len(I))][:n_remaining]
            _X[0, k, S] = True
        return _X


class MyMutation(Mutation):
    def _do(self, problem, X, **kwargs):
        for i in range(X.shape[0]):
            X[i, :] = X[i, :]
            is_false = np.where(np.logical_not(X[i, :]))[0]
            is_true = np.where(X[i, :])[0]
            X[i, np.random.choice(is_false)] = True
            X[i, np.random.choice(is_true)] = False
        return X

tdf=pd.read_csv("yml_nodups_labelled7.csv", sep='\t', encoding='utf-8')

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
               ('n_gen', 60),
               seed=1,
               verbose=True,
               save_history=True)

print("Function value: %s" % res.F[0])
indices = np.where(res.X)[0]
print("Subset:", list(indices))
pool.close()
sdf=tdf.iloc[indices]
sdf.to_csv("yml_small.csv", sep='\t', encoding='utf-8', index=False)