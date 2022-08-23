import requests
from datetime import datetime
import time
import io
import os
from ytb import YoutubeCommentDownloader
import csv
import pickle

'''
pickle.dump((None,0,None,0), open('checkpoint.chk', "wb" ))
pickle.load(open('checkpoint.chk', "rb"))

https://www.youtube.com/channel/UCaLSkUP3yReSALsFnbxAFKg
https://www.youtube.com/channel/channelID



https://youtube.googleapis.com/youtube/v3/commentThreads?part=snippet%2Creplies&allThreadsRelatedToChannelId=UC_x5XG1OV2P6uZZ5FSM9Ttw&key=AIzaSyAdr541s0uwQVcmfp8NkWlPufsD-7pcDEo


https://youtube.googleapis.com/youtube/v3/search?part=snippet&maxResults=500&type=video&key=AIzaSyAdr541s0uwQVcmfp8NkWlPufsD-7pcDEo&channelId=UCaLSkUP3yReSALsFnbxAFKg&order=rating

&channelId=UCaLSkUP3yReSALsFnbxAFKg
&order=viewCount
&key=AIzaSyAdr541s0uwQVcmfp8NkWlPufsD-7pcDEo
https://youtube.googleapis.com/youtube/v3/search?part=snippet&maxResults=500&type=video    
'''

checkpoint_file = 'checkpoint.chk'
api_key = None
api_keys = getKeys()
def getKeys():
    api_keys =['AIzaSyAdr541s0uwQVcmfp8NkWlPufsD-7pcDEo', 'AIzaSyC0d4hcvAk5-owjeUBpRJHEyk-vHcAG30k', 'AIzaSyA7KE_oR2AIc5ixGfMyEHDgxLz5FgWtZig', 'AIzaSyAu5ZCWWe4x5obWCx1Pe0TAgvf4PAdyQbU']
    for k in api_keys:
        yield k

def addUrl(url, key, it):
	if it:
		url+= "&"+key+"="+it
	return url

def checkResponse(res):
    if res.status_code != 200:
        print("Error for "+api_key+" url: "+res.url, end = " : ")
        print(res.json())
        return False
    return True

def getResponseCore(url, page=None, order = None):
    url = addUrl(url, "key", api_key)
	url = addUrl(url, "pageToken", page)
	url = addUrl(url, "order", order)
	return requests.get(url)

def getResponse(url, page=None, order = None):
    global api_key
    global api_keys
    res = None
    try:
        if api_key == None:
            api_key = next(api_keys)
        while True:
            res = getResponseCore(url, page, order)
            if checkResponse(res):
                break
            api_key = next(api_keys)
    except StopIteration:
        pass
	return res
	
def dumpCheckPoint(channel, videoPage):
    with open(checkpoint_file,"wb") as f:
        pickle.dump((channel, videoPage), f)

def getCheckPoint(clear=False):
    if not os.path.isfile(checkpoint_file):
        return (None,None)
    if clear:
        os.remove(checkpoint_file)
        return (None,None)
    channelPage, videoPage = (None,None)
    with open(checkpoint_file,"rb") as f:
        channelPage, videoPage = pickle.load(f)
    return (channelPage, videoPage)

def getThing(url, key, page, li, units, order):
    def newFunc():
        nonlocal url, key, page, li, units, order
        res = getResponse(url, page, order)
        if not checkResponse(res):
            yield (0, page)
            return
        res=res.json()
        cnt = 0
        i = 0
        while True:
            for it in res.get("items", []):
                if i >= li:
                    yield (it["id"][key], i, page)
                    cnt += 1
                if isinstance(units, int) and cnt >= units:
                    print("Max unit hit for "+key+" page: "+". Ending ...")
                    yield (i+1, page)
                i += 1
            page = res.get("nextPageToken")
            if page == None:
                yield (-1, page)
                break
            res = getResponse(url, page, order)
            if not checkResponse(res):
                yield (0, page)
                return
            res=res.json()
            i = 0
            li = 0            
    return newFunc

def getChannels(query="malayalam", page=None, li = 0, units = None, order="videoCount"):
    url = "https://youtube.googleapis.com/youtube/v3/search?part=snippet"
    url = addUrl(url, "maxResults", "500")
    url = addUrl(url, "Location", "India")
    url = addUrl(url, "regionCode", "IN")
    url = addUrl(url, "relevanceLanguage", "ml")
    url = addUrl(url, "type", "channel")
    url = addUrl(url, "q", query)
    return getThing(url, "channelId", page, li, units, order)

def getVideos(channelId, page=None, li = 0, units = None, order="rating"):
    url = "https://youtube.googleapis.com/youtube/v3/search?part=snippet"
    url = addUrl(url, "maxResults", "500")
    url = addUrl(url, "type", "video")
    url = addUrl(url, "channelId", channelId)
    url = addUrl(url, "order", order)
    res = getResponse(url, page, order)
    return getThing(url, "videoId", page, li, units, order)
		
def getComments(videoId, fp, cp,ci,vp,vi, header):
    downloader = YoutubeCommentDownloader()    
    gen = downloader.get_comments(videoId, language='ml')
    try:
        first = next(gen)
    except StopIteration:
        return
    first.update({'cp': cp, 'ci': ci, 'vp': vp, 'vi': vi})
    w = csv.DictWriter(fp, fieldnames=first.keys())
    if header:
        w.writeheader()
    w.writerow(first)
    for cmt in gen:
        cmt.update({'cp': cp, 'ci': ci, 'vp': vp, 'vi': vi})
        w.writerow(cmt)

def generate_timestamp():
    return time.strftime('%d_%b_%Y__%H_%M_%S', time.localtime(datetime.now().timestamp()))

def genException():
    raise ZeroDivisionError("You cannot divide a number by zero")

def main():
    targetCh, videoPage = (None,None)
    ch = None
    try:
        targetCh, videoPage = getCheckPoint()
        inputt = "channels_data.txt"
        output = "videos_data.txt"
        with io.open(output, 'a', encoding='utf8') as fp, io.open(inputt, 'r', encoding='utf8') as inp:
            print("\nBREAK\n",file=fp)
            flag1 = False
            for ch in inp:
                ch = ch.rstrip()
                if targetCh == None or ch == targetCh:
                    flag1=True
                if flag1:
                    for v in getVideos(ch, page=videoPage, li=vi)():    
                        print(ch, v, file=fp)
                        if isinstance(v[0], int):
                            if v[0] == -1:
                                break
                            if v[0] == 0:
                                dumpCheckPoint(ch, v[1]) 
                                print("Quota Exhausted all list of videos. Ending ...")
                                exit(0)
                            exit(1)
    except Exception as e:
        print(e)
        dumpCheckPoint(ch, videoPage) 

if __name__ == '__main__':
	main()