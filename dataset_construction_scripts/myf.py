from random import randrange
import multiprocessing
import os
from time import sleep
'''
def testf(x):
 t=randrange(10)
 print("Proccess id: ", os.getpid(), " t ", t)
 #sleep(3)
 if t > 5:
  print("raising")
  raise Exception
 print("printing "+ str(x))
 return x+1
'''

#from deep_translator import GoogleTranslator as gt
from time import sleep
import requests
from lxml import html as ht

def mlConv(row):
 mltext = row['text']
 res=requests.get('https://translate.google.com/m?sl=ml&tl=en&hl=en&q='+mltext)
 if res.status_code != 200:
  raise Exception
 tr=ht.fromstring(res.content)
 row['conv-text'] = tr.xpath('//div[contains(@class, "result-container")]')[0].text
 return row

df.apply(mlConv, axis=1)
'''
 ml2e=gt(source="ml",target="en")
 try:
  row['conv-text'] = ml2e.translate(row['text'])
 except:
  print("transl err")
  raise
 return row
 '''
 