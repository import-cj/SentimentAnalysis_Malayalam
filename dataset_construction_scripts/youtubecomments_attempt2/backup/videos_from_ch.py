from pytube import Channel
from ytb import YoutubeCommentDownloader
import csv

import requests as r
from lxml import html as ht
import os

cookies = {
    '_ga': 'GA1.2.1457901287.1655117636',
    '_gid': 'GA1.2.258819648.1655117636',
    'CAKEPHP': '4u97shojai7hb9b9cmvm5cpq2n',
    '_gat': '1',
}

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:100.0) Gecko/20100101 Firefox/100.0',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-GB,en;q=0.5',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'same-origin',
    'Sec-Fetch-User': '?1',
}

def getComments_(videoId, fp, header):
    downloader = YoutubeCommentDownloader()    
    gen = downloader.get_comments(videoId, language='ml')
    try:
        first = next(gen)
    except StopIteration:
        return
    w = csv.DictWriter(fp, fieldnames=first.keys())
    if header:
        w.writeheader()
    w.writerow(first)
    for cmt in gen:
        w.writerow(cmt)
    fp.flush()

def getComments():
    for path,file in getPaths('videos', 'comments'):
        header = True
        with open(path) as videos, open(os.path.join("comments",file), "a") as comments:
            for v in videos:
                getComments_(v, comments, header)
                header = False

def getChannelVideos(ch, fp):
    c = Channel('https://www.youtube.com/channel/'+ch)
    for v in c.videos:
        print(v.video_id, file=fp)

def getPaths(folder, suff):
    def append_id(filename, suff):
        return "{0}_{2}{1}".format(*os.path.splitext(filename) + (suff,))
    for file in os.listdir(folder): # Find all files in the input folder
        yield file, append_id(file, suff)

def getVideos():
    for path,file in getPaths('channels', 'videos'):
        with open(path) as channels, open(os.path.join("videos",file), "a") as videos:
            for ch in channels:
                getChannelVideos(ch, videos)

def getData(cat='', lang=''):
	data = [
		('_method', 'POST'),
		('data[query][name]', ''),
		('dropdownsearch', cat),
		('dropdownsearch', lang),
		('data[query][min_subs]', ''),
		('data[query][max_subs]', ''),
		('data[query][min_views]', ''),
		('data[query][max_views]', ''),
		('data[query][min_videos]', ''),
		('data[query][max_videos]', ''),
		('data[query][min_published_on]', ''),
		('data[query][max_published_on]', ''),
		('query[category_id][]', cat),
		('query[language_id][]', lang),
		('query[category_id][]', cat),
		('query[language_id][]', lang),
	]

def begin(cat, lang):
	response = r.post('https://channelcrawler.com/eng', cookies=cookies, headers=headers, data=getData(cat, lang))
	tr=ht.fromstring(response.content)
	pages = max(len(tr.xpath('//nav[@aria-label="Page navigation"]//li')) - 2, 0)
	return response.url, pages

def getChannels_(urls, pages, cat, catn):
    print(catn, pages)
    for url in urls:
        for i in range(1,pages+1):
            res=r.get(url+str(i), cookies=cookies, headers=headers)
            print(res.url, res.status_code)
            tr=ht.fromstring(res.content)
            with open(os.path.join("channels", "channelList_"+cat+"_"+catn+".txt"),"a") as f:
                for d in tr.xpath('/html/body/div[1]/div[1]/div/div[2]/div/a/@href'):
                    print(d.split("/")[-1], file=f)

def getUrls(url, pages):
    if pages < 13:
        yield url+'/sort:Channel2.subscribers/direction:desc/page:'
        return
    urls = [
    '/sort:Channel2.subscribers/direction:desc/page:',
    '/sort:Channel2.subscribers/direction:asc/page:',
    '/sort:Channel2.videos/direction:desc/page:',
    '/sort:Channel2.videos/direction:asc/page:',
    '/sort:Channel2.views/direction:desc/page:',
    '/sort:Channel2.views/direction:asc/page:',
    '/sort:Channel2.published_on/direction:asc/page:',
    '/sort:Channel2.published_on/direction:desc/page:',
    '/sort:Channel2.last_video_date/direction:asc/page:',
    '/sort:Channel2.last_video_date/direction:desc/page:'
    ]
    for u in urls:
        yield url+u

def getChannels():
    cats = {"23":"Comedy", "27":"Education", "24":"Entertainment", "2":"Autos_and_Vehicles", "1":"Film_and_Animation", "20":"Gaming", "26":"Howto_and_Style", "10":"Music", "25":"News_and_Politics", "29":"Nonprofits_and_Activism", "22":"People_and_Blogs", "15":"Pets_and_Animals", "28":"Science_and_Technology", "17":"Sports", "19":"Travel_and_Events"}
    lang = "419"
    for cat in cats:
        base, pages = begin(cat, lang)
        getChannels_(getUrls(base, pages), pages, cat, cats[cat])

def getAll():
    pass

def init(path):
    isExist = os.path.exists(path)
    if isExist:
        print("Error folder exists : "+ path)
        exit(1)
    os.makedirs(path)

def main():
    init("videos")
    init("channels")
    init("comments")
    getChannels() # store it as files in Channels folder
    print("channels downloaded")
    getVideos() # store it as files in Videos folder
    print("videos downloaded")
    getComments() # store it as files in Comments folder
    getAll() # store a merged files of all comments, comments.csv

if __name__ == "__main__":
    main()