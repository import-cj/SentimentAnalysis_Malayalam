from deep_translator import GoogleTranslator as gt
from time import sleep

def mlConvEn(text):
 ml2e=gt(source="ml",target="en") 
 eTxt=None
 st=0.1
 while True:
  try:
   eTxt = ml2e.translate(text)
  except:
   eTxt=None
   print("Sleeping for "+str(st))
   sleep(st)
   ml2e=gt(source="ml",target="en") 
   st+=st
   if st > 100:
    st=0.1
   continue
  break
 return text