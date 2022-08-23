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

api_key = 'AIzaSyAdr541s0uwQVcmfp8NkWlPufsD-7pcDEo'
checkpoint_file = 'checkpoint.chk'
def addUrl(url, key, it):
	if it:
		url+= "&"+key+"="+it
	return url

def getResponse(url, page=None, order = None):
	url = addUrl(url, "key", api_key)
	url = addUrl(url, "pageToken", page)
	url = addUrl(url, "order", order)
	return requests.get(url)
	
def dumpCheckPoint(channelPage, ci, videoPage, vi):
    with open(checkpoint_file,"wb") as f:
        pickle.dump((channelPage, ci, videoPage, vi), f)

def getCheckPoint(clear=False):
    if not os.path.isfile(checkpoint_file):
        return (None,0,None,0)
    if clear:
        os.remove(checkpoint_file)
        return (None,0,None,0)
    channelPage, ci, videoPage, vi = (None,0,None,0)
    with open(checkpoint_file,"rb") as f:
        channelPage, ci, videoPage, vi = pickle.load(f)
    return (channelPage, ci, videoPage, vi)
    
def getThing(url, key, page, li, units, order):
    def newFunc():
        nonlocal url, key, page, li, units, order
        res = getResponse(url, page, order).json()
        cnt = 0
        i = 0
        while True:
            for it in res["items"]:
                if i >= li:
                    yield (it["id"][key], i, page)
                    cnt += 1
                if isinstance(units, int) and cnt >= units :
                    print("Max unit hit for "+key+" page: "+". Ending ...")
                    yield (i+1, page)
                i += 1
            page = res.get("nextPageToken")
            if page == None:
                yield (-1, page)
                break
            res = getResponse(url, page)
            if res.status_code != 200:
                print("Error for "+key+" page: "+page, end = " : ")
                print(res["error"]["message"])
                yield (0, page)
            i = 0
            li = 0
            res=res.json()
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
    channelPage, ci, videoPage, vi = (None,0,None,0)
    ch = (channelPage, ci)
    v  = (videoPage, vi)
    try:
        channelPage, ci, videoPage, vi = getCheckPoint()
        output = "channels_data_"+generate_timestamp()+".txt"
        with io.open(output, 'w', encoding='utf8') as fp:
            i=0
            for ch in getChannels(page=channelPage, li=ci)():
                if isinstance(ch[0], int):
                    if ch[0] == -1:
                        print("Exhausted all list of channels. Ending ...")
                        exit(0)
                    exit(1)
                print(ch, file=fp)
                i+=1
                if i==10000:
                    fp.flush()
                    i=0
    except Exception as e:
        print(e)
        #dumpCheckPoint(ch[0], ch[1], v[0], v[1])
        

if __name__ == '__main__':
	main()