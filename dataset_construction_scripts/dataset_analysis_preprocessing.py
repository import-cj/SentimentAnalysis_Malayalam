'''
>>> e2ml.translate("insult")
'അപമാനം'
>>> e2ml.translate("investment")
'നിക്ഷേപം'
>>> e2ml.translate("INSULT ആണ് മുരളീ ലോകത്തിലെ ഏറ്റവും വലിയ INVESTMENT")
'INSULT ആണ് മുരളീ ലോകത്തിലെ ഏറ്റവും വലിയ INVESTMENT'
>>> bb="അപമാനം ആണ് മുരളീ ലോകത്തിലെ ഏറ്റവും വലിയ നിക്ഷേപം"
>>> ml2e.translate(bb)
"Insult is Murali's biggest investment in the world"
>>> e2ml.translate("baffalo")
'ബഫല്ലോ'
>>> ml2e.translate(e2ml.translate("baffalo"))
'Buffalo'
>>> e2ml.translate(ml2e.translate(e2ml.translate("baffalo")))
'പോത്ത്'
'''
from deep_translator import GoogleTranslator as gt
e2ml=gt(source="en",target="ml") 

import pandas as pd
import unicodedata

def acceptChar(c):
 # space, comma
 if c==' ' or c=='\t' or c==',':
  return False
 # ml
 if u'\u0d01' <= c <= u'\u0d7f':
  return False
 cat=unicodedata.category(c)
 nam=''
 try:
  nam=unicodedata.name(c)
 except:
  pass
 # numbers
 if cat == 'Nd' and nam.startswith("DIGIT"):
  return False
 # currency
 if cat == 'Sc' and (nam == "INDIAN RUPEE SIGN" or nam == "DOLLAR SIGN"):
  return False
 if cat == 'Sm':
  return False
 # rest of puncutations
 if cat[0] == 'P':
  return False
 return True

def textFilter(row):
 text = row['text']
 for c in text:
  if acceptChar(c):
   return False
 return True

df = pd.read_csv("youtube_comments_text7.csv", sep='\t', encoding='utf-8')
mask = df.apply(textFilter, axis=1)

df1 = df[mask]
df2 = df[~mask]

df1.to_csv("youtube_ml_comments_pure.csv", sep='\t', encoding='utf-8', index=False)
df2.to_csv("youtube_ml_comments_mixd.csv", sep='\t', encoding='utf-8', index=False)




def mlConvEn(text):
 ml2e=gt(source="ml",target="en") 
 eTxt=None
 st=0.1
 while True:
  try:
   eTxt = ml2e.translate(text)
  except:
   eTxt=None
   #print("Sleeping for "+str(st))
   sleep(st)
   ml2e=gt(source="ml",target="en") 
   st+=st
   if st > 100:
    st=0.1
   continue
  break
 return row

import pandas as pd
df1=pd.read_csv("youtube_ml_comments_pure.csv", sep='\t', encoding='utf-8')
import multiprocessing as mp
from mlconv import *

def doit(dft):
 with mp.Pool(mp.cpu_count()) as pool:
  dft['conv-text'] = pool.map(mlConvEn, dft['text'])
 dft.to_csv("youtube_ml_comments_pure_enConv.csv", sep='\t', encoding='utf-8', index=False)

doit(df1)


from myf import *
import multiprocessing as mp
with mp.Pool(mp.cpu_count()) as pool:
  pool.map(testf, range(10))

from myf import *
import multiprocessing as mp
with mp.Pool(mp.cpu_count()) as pool:
 results = {}
 for i in range(10):
   results[i] = pool.apply_async(testf,(i,))
 run=True
 while run:
  restarts=[]
  run=False
  for func, result in results.items():
   if result.ready():
    if not result.successful():
     restarts.append(func)
     run=True
     continue
    continue
   run=True
  for i in restarts:
   results[i] = pool.apply_async(testf,(i,))
 for func, result in results.items():
  print(func, result.get())


from time import sleep
import pandas as pd
from myf import *
import multiprocessing as mp
df = pd.read_csv("youtube_ml_comments_rest.csv", sep='\t', encoding='utf-8')
df = df.reset_index()
with mp.Pool(mp.cpu_count()) as pool, open("youtube_ml_comments_pure_enConv_live.csv", "a",  encoding='utf8') as fp:
 results = []
 run=True
 genr = df.iterrows()
 while len(results) < 25:
   try:
    i,row = next(genr)
    results.append((pool.apply_async(mlConv,(row,)),row))
    #print("add new")
   except StopIteration:
    break
 while len(results) != 0:
  restarts=[]
  run=False
  sleepytime = 10
  for i in range(len(results)):
   result=results[i][0]
   if result.ready():
    if not result.successful():
     restarts.append(results[i][1])
     results[i] = None
     run=True
     #print("not success")
     continue
    print(result.get().to_csv(header=False, index=False, sep='\t', encoding='utf-8',line_terminator='\t'),file=fp)
    results[i] = None
    sleepytime = 0
    #print("success")
    continue
   run=True
   #print("not ready")
  results = [k for k in results if k is not None]
  sleep(sleepytime)  
  for row in restarts:
   results.append([pool.apply_async(mlConv,(row,)),row])
  while len(results) < 25:
   try:
    i,row = next(genr)
    results.append((pool.apply_async(mlConv,(row,)),row))
    #print("add new")
   except StopIteration:
    break


import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob')

def fun1(s):
 doc = nlp(s)
 sentiment = doc._.blob.polarity
 return round(sentiment,2)


from monkeylearn import MonkeyLearn
ml = MonkeyLearn('64b0da72e9668e304eee59a9b8014b1ee31cdfc1')
data = [
"Dear Arun, I respect you. I can only look at you with respect. Thank you for trying to bring us the culture and heritage of the respective countries using your humble technology.",
"No, Khamar Roosh is a Maoist-minded sect whose leader was Pol Pot, and he was the cause of this massacre.",
"Is the video shot on a mobile phone ? Try to put the videos in the same order . If you put the new one and the old one like this , the viewers will not understand anything . The previous videos are very clear . This one is not very clear"
]
model_id = 'cl_pi3C7JiL'
result = ml.classifiers.classify(model_id, data)
print(result.body)


from collections import defaultdict
import pandas as pd
pd.set_option("display.max_rows", None)
df=pd.read_csv("youtube_ml_comments_pure.csv", sep='\t', encoding='utf-8')
cd=defaultdict(int)
with open("log.txt", 'w', encoding='utf-8') as fp:
 print(df.text.str.split(expand=True).stack().value_counts(),file=fp)

df=pd.read_csv("youtube_ml_comments_mixd.csv", sep='\t', encoding='utf-8')
cd=defaultdict(int)
with open("logmix.txt", 'w', encoding='utf-8') as fp:
 print(df.text.str.split(expand=True).stack().value_counts(),file=fp)


import time
import datetime

df=pd.read_csv("youtube_ml_comments_pure_enConv_live.csv", sep='\t', encoding='utf-8', usecols=[2,7,8], names=['m','c','e'])





from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

sdf=df.sample(frac=0.10)
l=list(sdf['m'])
start = time.perf_counter()
sentence_embeddings = model.encode(l)
end = time.perf_counter()
print(str(datetime.timedelta(seconds=end - start)))

from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity(
    [sentence_embeddings[0]],
    sentence_embeddings[1:]
)
t=[ l[i+1] for i in range(len(sim[0])) if sim[0][i] > 0.92]
t=[l[0]]+t
with open("cosim.txt", 'w', encoding='utf-8') as fp:
 print(t,file=fp)

import pandas as pd
tdf=pd.read_csv("youtube_ml_comments_pure_enConv_live.csv", sep='\t', encoding='utf-8')
def sort_df(df, column_idx, key):
    column_idx=df.columns.get_loc(column_idx) 
    col = df.iloc[:,column_idx]
    print(col)
    df = df.iloc[[i[1] for i in sorted(zip(col,range(len(col))), key=key)]]
    return df

tdf=tdf.drop_duplicates(keep=False)
cmp = lambda x:len(x)
ndf=sort_df(tdf,'Entext',cmp).drop_duplicates('cid', keep='last')

def dups(df):
 ids = df["cid"]
 print(df[ids.isin(ids[ids.duplicated()])].sort_values("cid"))

df2.to_csv("yml_nodups.csv", sep='\t', encoding='utf-8', index=False)

col = df.ix[:,column_idx]
df = df.ix[[i[1] for i in sorted(zip(col,range(len(col))), key=key)]]

import pandas as pd
tdf=pd.read_csv("yml_nodups.csv", sep='\t', encoding='utf-8')
tdf.Entext=tdf.Entext.fillna('')
import pandas as pd
from flair.models import TextClassifier
from flair.data import Sentence
def flairSent(model):
 sentiment_model = TextClassifier.load(model)
 def fun2(s):
  nonlocal sentiment_model
  if s=='':
   return s
  try:
   sentence = Sentence(s)
   sentiment_model.predict(sentence)
   l=sentence.labels
   if len(l)==0:
    print(s)
    return ""
   l=l[0]
  except:
   print(s)
   raise
  return l.value,l.score
 return fun2


from pysentimiento import create_analyzer
def pysent(model):
 analyzer = create_analyzer(task=model, lang="en")
 def fun2(s):
  nonlocal analyzer
  if s=='':
   return s
  ret=""
  try:
   t = analyzer.predict(s)
   if isinstance(t.output, list):
    for it in t.output:
     ret=it+" "
   else:
    ret+=t.output
   ret+=":"
   for k in t.probas:
    if t.probas[k]> 0.3:
     ret= ret+" "
  except:
   print(s)
   raise
  return ret
 return fun2

def addStuff(df,fn, model, label):
 cbk = fn(model)
 df[label] = df.apply (lambda row: cbk(row['Entext']), axis=1)
 return df

tdf=addStuff(tdf,flairSent, "sentiment-fast", "flair_sent_fast")
tdf.to_csv("yml_nodups_labelled1.csv", sep='\t', encoding='utf-8', index=False)

tdf=addStuff(tdf,flairSent, "en-sentiment", "flair_sent_en")
tdf.to_csv("yml_nodups_labelled2.csv", sep='\t', encoding='utf-8', index=False)

import pandas as pd
tdf=pd.read_csv("yml_nodups_labelled2.csv", sep='\t', encoding='utf-8')
tdf.Entext=tdf.Entext.fillna('')
tdf=addStuff(tdf,pysent, "sentiment", "pysent_sent")
tdf.to_csv("yml_nodups_labelled3.csv", sep='\t', encoding='utf-8', index=False)

tdf=addStuff(tdf,pysent, "emotion", "pysent_emot")
tdf.to_csv("yml_nodups_labelled4.csv", sep='\t', encoding='utf-8', index=False)

tdf=addStuff(tdf,pysent, "hate_speech", "pysent_hate")
tdf.to_csv("yml_nodups_labelled5.csv", sep='\t', encoding='utf-8', index=False)



import pandas as pd
pd.set_option("display.max_rows", None)

def swrds(n):
 df=pd.read_csv("youtube_comments_text7.csv", sep='\t', encoding='utf-8')
 df=df[df['text'].str.split(' ').apply(len) < n]
 print(len(df))
 with open("swrds"+str(n)+".txt", 'w', encoding='utf-8') as fp:
  print(df.text.str.split(expand=True).stack().value_counts(),file=fp)

from profanity_check import predict
from better_profanity import profanity
from profanity_filter import ProfanityFilter
import pandas as pd
tdf=pd.read_csv("yml_nodups_labelled5.csv", sep='\t', encoding='utf-8')
tdf.Entext=tdf.Entext.fillna('')

def prof():
 pf = ProfanityFilter()
 def fun2(s):
  nonlocal pf
  if s=='':
   return 0
  if profanity.contains_profanity(s) or  predict([s])[0]==1 or pf.is_profane(s):
   return 1
  return 0
 return fun2

def addStuff(df,fn):
 cbk = fn()
 df["offense"] = df.apply (lambda row: cbk(row['Entext']), axis=1)
 return df

tdf=addStuff(tdf,prof)
tdf.to_csv("yml_nodups_labelled6.csv", sep='\t', encoding='utf-8', index=False)

df=pd.read_csv("log.txt", sep='\t', encoding='utf-8',header=None,names=['t','n'])
mask1 = (df['n']<10) & (df['t'].str.len() < 10)
mask2 = (df['n']>6)
mask3 = (df['t'].str.len() < 6)
mask4 = (df['t'].str.len() < 20)
lng=df[mask]
srt=df[~mask]

def getit(tt):
 def doit(row):
  nonlocal tt
  for w in tt['t']:
   row['t']=row['t'].replace(w,w+" ")
  return row
 return doit
 
lng=lng.apply(getit(srt),axis=1)
lng.to_csv("fixwords.csv", sep='\t', encoding='utf-8', index=False)


(([^\u0d00-\u0d7f\s\da-zA-Z])\2{1,})

import re
import requests
from lxml import html as ht
from better_profanity import profanity
from profanity_filter import ProfanityFilter
import pandas as pd
from flair.models import TextClassifier
from flair.data import Sentence
from pysentimiento import create_analyzer
tdf=pd.read_csv("yml_nodups_labelled6.csv", sep='\t', encoding='utf-8')
tdf.Entext=tdf.Entext.fillna('')

d={}
def flairSent(model):
 global d
 if model not in d:
  sentiment_model = TextClassifier.load(model)
  d[model]=sentiment_model
 sentiment_model = d[model]
 def fun2(s):
  nonlocal sentiment_model
  if s=='':
   return s
  try:
   sentence = Sentence(s)
   sentiment_model.predict(sentence)
   l=sentence.labels
   if len(l)==0:
    print(s)
    return ""
   l=l[0]
  except:
   print(s)
   raise
  return l.value,l.score
 return fun2

def pysent(model):
 global d
 if model not in d:
  analyzer = create_analyzer(task=model, lang="en")
  d[model]=analyzer
 analyzer = d[model]
 def fun2(s):
  nonlocal analyzer
  if s=='':
   return s
  ret=""
  try:
   t = analyzer.predict(s)
   if isinstance(t.output, list):
    for it in t.output:
     ret=it+" "
   else:
    ret+=t.output
   ret+=":"
   for k in t.probas:
    if t.probas[k]> 0.3:
     ret= ret+" "
  except:
   print(s)
   raise
  return ret
 return fun2

def addStuff(row,fn, model,label):
 cbk = fn(model)
 row[label]=cbk(row['Entext'])
 return row

def mlConv(row):
 res=requests.get('https://translate.google.com/m?sl=ml&tl=en&hl=en&q='+row['text'])
 if res.status_code != 200:
  raise Exception
 tr=ht.fromstring(res.content)
 row['Entext'] = tr.xpath('//div[contains(@class, "result-container")]')[0].text
 return row

def rgxrpl(row):
    global sentiment_model
    global analyzer
    s=row['text']
    found = False
    def inner(mtc):
        nonlocal found
        found=True
        return ' '
    pat = re.compile(r'(([^\u0d00-\u0d7f\s\da-zA-Z])\2{1,})',re.UNICODE)
    s=re.sub(pat, inner, s)
    if not found:
        return row
    row['text']=s
    row=mlConv(row)
    pf = ProfanityFilter()
    row['offense'] = 0
    s=row['Entext']
    if profanity.contains_profanity(s) or pf.is_profane(s):
        row['offense'] = 1
    row=addStuff(row,flairSent, "sentiment-fast", "flair_sent_fast")
    row=addStuff(row,flairSent, "en-sentiment", "flair_sent_en")
    row=addStuff(row,pysent, "sentiment", "pysent_sent")
    row=addStuff(row,pysent, "emotion", "pysent_emot")
    row=addStuff(row,pysent, "hate_speech", "pysent_hate")
    return row

tdf=tdf.apply(rgxrpl,axis=1)
tdf.to_csv("yml_nodups_labelled7.csv", sep='\t', encoding='utf-8', index=False)

tdf=pd.read_csv("yml_nodups_labelled7.csv", sep='\t', encoding='utf-8')
df=pd.read_csv("log.txt", sep='\t', encoding='utf-8',header=None,names=['t','n'])
m1 = (df['n']<10) & (df['t'].str.len() < 16)
m2 = (df['n']>5)
m3 = (df['t'].str.len() < 6)
m4 = (df['t'].str.len() < 25)
m5 = m1|m2|m3|m4
df = df[~m5]
df.to_csv("removelog.txt", sep='\t', encoding='utf-8', index=False)
wors=list(df['t'])

def tdffilter(row):
 global wors
 y=row['text'].split()
 for w in wors:
  if w in y:
   return False  
 return True

tdf=pd.read_csv("yml_nodups_labelled7.csv", sep='\t', encoding='utf-8')
tdf=tdf[tdf.apply(tdffilter, axis=1)]
tdf.to_csv("yml_nodups_labelled8.csv", sep='\t', encoding='utf-8', index=False)