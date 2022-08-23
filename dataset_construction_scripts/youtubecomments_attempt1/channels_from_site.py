import requests as r
from lxml import html as ht

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
		('query[language_id][]', lang`),
	]

def getUrls(url, pages):
    if pages < 13:
        return [url+'/sort:Channel2.subscribers/direction:desc/page:']
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


def begin(cat, lang):
	response = requests.post('https://channelcrawler.com/eng', cookies=cookies, headers=headers, data=getData(ct, lang))
	tr=ht.fromstring(response.content)
	pages = max(len(tr.xpath('//nav[@aria-label="Page navigation"]//li')) - 2, 0)
	return response.url, pages

def getChannels(urls, pages, cat):
    for url in urls:
        for i in range(1,pages+1):
            res=r.get(url+str(i), cookies=cookies, headers=headers)
            print(res.url, res.status_code)
            tr=ht.fromstring(res.content)
            with open("channelList_cat_"+cat+".txt","a") as f:
                for d in tr.xpath('/html/body/div[1]/div[1]/div/div[2]/div/a/@href'):
                    print(d.split("/")[-1], file=f)

def main():
	cats = ["23","27","24","2","1","20","26","10","25","29","22","15","28","17","19"]
	lang = "419"
    for cat in cats:
        base, pages = begin(cat, lang)
        getChannels(getUrls(base, pages), pages, cat)