from collections import deque
import pickle
import pandas as pd
import numpy as np
import functools

column="text"

def gettotals(df):
 chars = set()
 results = set()
 df[column].str.split().apply(results.update) 
 for w in results:
  chars.update(w)
 #print(len(chars),len(results))
 return chars,results

def printStats(df):
 charlist = []
 wordlist = []
 for i,row in df.iterrows():
  words = set(row[column].split())
  wordlist.append(words)
  chars = set()
  for w in words:
   chars.update(w)
  charlist.append(chars)
 return charlist,wordlist

def getNextIdx(df, idx ,cidx, chars, words):
 s1 = np.setdiff1d(idx, cidx, assume_unique=True)
 s2 = np.setdiff1d(cidx, idx, assume_unique=True)
 s  = np.concatenate((s1,s2))
 if len(s) == 0:
  print("same")
  #print(idx)
  return idx,0
 s3 = np.intersect1d(cidx, idx, assume_unique=True)
 c,w=gettotals(df.iloc[s3])
 l=[]
 #print(s1,s2,s,s3)
 #print(len(df))
 #print(s)
 for i in range(s.shape[0]):
  #print(i)
  ind = s[i]
  #print(ind)
  t1=len(c.union(chars[ind]))
  #print(i)
  t2=len(w.union(words[ind]))
  #print(i)
  l.append((t1, t2, s[i]))
 swaps = 0
 def make_comparator(less_than):
  def compare(x, y):
   if less_than(x, y):
    return -1 #-1
   elif less_than(y, x):
    return 1  #1
   else:
    return 0
  return compare
 def lessthan(x,y):
  nonlocal chars
  nonlocal words
  nonlocal swaps
  if x == y:
   return False
  #if x[0] - y[0] > 2 or (x[0] - y[0] > 0 and x[1] - y[1] > 10):   
  if x[0] - y[0] > 0 or (x[0] - y[0] >= 0 and x[1] - y[1] > 4) or x[1] - y[1] > 3:
   swaps+=1
   return True
  return False
 l = sorted(l, key=functools.cmp_to_key(make_comparator(lessthan)))
 k = idx.shape[0] - s3.shape[0]
 return (np.append(s3, [ l[i][2] for i in range(k)]),swaps)
   

np.random.seed(73)

def randSample(mx, n):
 return np.random.choice(mx, n, replace = False)

'''
def randSampleMutateOne(mx, idx, df, chars):
 c,w=gettotals(df.iloc[idx])
 while True:
  m = randSample(mx,1)
  if m in idx:
   continue
  chars[m]
  return idx
 return None
'''

def getCharRows(chars, chs):
 for i in range(len(chars)):
  for ch in chars[i]:
   if ch not in chs:
    yield i
    break

def getDupilcate(chars, idx):
 #print(idx)
 for i in range(len(idx)):
  for j in range(len(idx)):
   if i <=j:
    continue
   #print(i,j, idx)
   seti = chars[idx[i]]
   setj = chars[idx[j]]
   yy = seti.intersection(setj)
   if len(seti) - len(yy) < 4:
    yield i
   if len(setj) - len(yy) < 4:
    yield j

def randSampleNew(tdf, chars, idx):
 #print("before")
 #print(idx)
 _,t1 = gettotals(tdf.iloc[idx])
 for i in getCharRows(chars, t1):
  for j in getDupilcate(chars, idx):
   #print(idx)
   #print(i,j)
   #print(i in idx)
   #print(j in idx)
   oidx = idx.copy()
   oidx[j] = i
   #print("found")
   #print(np.setdiff1d(oidx, idx, assume_unique=True))
   return oidx
 return idx


def doit(tdf, chars, words, n_samples, idx, l):
 #print(idx)
 a,b = gettotals(tdf.iloc[idx])
 a=len(a)
 b=len(b)
 while True:
   cidx = randSample(len(tdf), n_samples)
   #print(idx)
   #cidx = randSampleNew(tdf, words, idx)
   #print(np.setdiff1d(cidx, idx, assume_unique=True))
   idx,swaps = getNextIdx(tdf, idx ,cidx, chars, words)
   #print(idx)
   if swaps > 0:
    c,d = gettotals(tdf.iloc[idx])
    if len(c)>a or len(d)>b-30:
     a=len(c)
     b=len(d)
     analyse(tdf, idx)
     l.append(idx)

def analyse(tdf, idx):
 t1,t2 = gettotals(tdf.iloc[idx])
 print("added ",len(t1), len(t2))

def getnpsort(df):
 def npsort1(x):
  nonlocal df
  for idx in x:
   t1,t2 = gettotals(tdf.iloc[idx])  
   yield len(t1)
 return npsort1

def main():
 tdf = pd.read_csv("neut.csv", sep='\t', encoding='utf-8')
 n_samples = 1500
 pops = None
 idx=None
 try:
  pops = np.load('pops.npy', allow_pickle=True)
  idx = pops[-1,]
  idx = idx[:n_samples]
  pops = np.array([idx])
 except:
  idx = randSample(len(tdf),n_samples)
  pops = np.array([idx])
 chars,words = printStats(tdf)
 l = []
 try:
  doit(tdf, chars, words, n_samples, idx, l)
 except KeyboardInterrupt:
  pops = np.concatenate((pops,l))
  pass 
 np.save('pops.npy', pops, allow_pickle=True)
 print('interrupted!')
 print(len(pops))
 return pops

t=main()