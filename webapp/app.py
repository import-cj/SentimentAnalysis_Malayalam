# Contents of ~/my_app/streamlit_app.py
import streamlit as st

import json
import re
import time

import dateparser
import requests

try:
    from streamlit.script_run_context import get_script_run_ctx
except ModuleNotFoundError:
    # streamlit < 1.4
    from streamlit.report_thread import (  # type: ignore
        get_report_ctx as get_script_run_ctx,
    )

from streamlit.server.server import Server

# Ref: https://gist.github.com/tvst/036da038ab3e999a64497f42de966a92


def get_session_id() -> str:
    ctx = get_script_run_ctx()
    if ctx is None:
        raise Exception("Failed to get the thread context")

    return ctx.session_id


def get_this_session_info():
    current_server = Server.get_current()

    # The original implementation of SessionState (https://gist.github.com/tvst/036da038ab3e999a64497f42de966a92) has a problem    # noqa: E501
    # as referred to in https://gist.github.com/tvst/036da038ab3e999a64497f42de966a92#gistcomment-3484515,                         # noqa: E501
    # then fixed here.
    # This code only works with streamlit>=0.65, https://gist.github.com/tvst/036da038ab3e999a64497f42de966a92#gistcomment-3418729 # noqa: E501
    session_id = get_session_id()
    session_info = current_server._get_session_info(session_id)

    return session_info



YOUTUBE_VIDEO_URL = 'https://www.youtube.com/watch?v={youtube_id}'

USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36'

SORT_BY_POPULAR = 0
SORT_BY_RECENT = 1

YT_CFG_RE = r'ytcfg\.set\s*\(\s*({.+?})\s*\)\s*;'
YT_INITIAL_DATA_RE = r'(?:window\s*\[\s*["\']ytInitialData["\']\s*\]|ytInitialData)\s*=\s*({.+?})\s*;\s*(?:var\s+meta|</script|\n)'


class YoutubeCommentDownloader:

    def __init__(self):
        self.session = requests.Session()
        self.session.headers['User-Agent'] = USER_AGENT

    def ajax_request(self, endpoint, ytcfg, retries=5, sleep=20):
        url = 'https://www.youtube.com' + endpoint['commandMetadata']['webCommandMetadata']['apiUrl']

        data = {'context': ytcfg['INNERTUBE_CONTEXT'],
                'continuation': endpoint['continuationCommand']['token']}

        for _ in range(retries):
            response = self.session.post(url, params={'key': ytcfg['INNERTUBE_API_KEY']}, json=data)
            if response.status_code == 200:
                return response.json()
            if response.status_code in [403, 413]:
                return {}
            else:
                time.sleep(sleep)

    def get_comments(self, youtube_id, *args, **kwargs):
        return self.get_comments_from_url(YOUTUBE_VIDEO_URL.format(youtube_id=youtube_id), *args, **kwargs)

    def get_comments_from_url(self, youtube_url, sort_by=SORT_BY_RECENT, language=None, sleep=.1):
        response = self.session.get(youtube_url)

        if 'uxe=' in response.request.url:
            self.session.cookies.set('CONSENT', 'YES+cb', domain='.youtube.com')
            response = self.session.get(youtube_url)

        html = response.text
        ytcfg = json.loads(self.regex_search(html, YT_CFG_RE, default=''))
        if not ytcfg:
            return  # Unable to extract configuration
        if language:
            ytcfg['INNERTUBE_CONTEXT']['client']['hl'] = language

        data = json.loads(self.regex_search(html, YT_INITIAL_DATA_RE, default=''))

        section = next(self.search_dict(data['contents'], 'itemSectionRenderer'), None)
        renderer = next(self.search_dict(section, 'continuationItemRenderer'), None) if section else None
        if not renderer:
            # Comments disabled?
            return

        needs_sorting = sort_by != SORT_BY_POPULAR
        continuations = [renderer['continuationEndpoint']]
        while continuations:
            continuation = continuations.pop()
            response = self.ajax_request(continuation, ytcfg)

            if not response:
                break

            error = next(self.search_dict(response, 'externalErrorMessage'), None)
            if error:
                raise RuntimeError('Error returned from server: ' + error)

            if needs_sorting:
                sort_menu = next(self.search_dict(response, 'sortFilterSubMenuRenderer'), {}).get('subMenuItems', [])
                if sort_by < len(sort_menu):
                    continuations = [sort_menu[sort_by]['serviceEndpoint']]
                    needs_sorting = False
                    continue
                raise RuntimeError('Failed to set sorting')

            actions = list(self.search_dict(response, 'reloadContinuationItemsCommand')) + \
                      list(self.search_dict(response, 'appendContinuationItemsAction'))
            for action in actions:
                for item in action.get('continuationItems', []):
                    if action['targetId'] == 'comments-section':
                        # Process continuations for comments and replies.
                        continuations[:0] = [ep for ep in self.search_dict(item, 'continuationEndpoint')]
                    if action['targetId'].startswith('comment-replies-item') and 'continuationItemRenderer' in item:
                        # Process the 'Show more replies' button
                        continuations.append(next(self.search_dict(item, 'buttonRenderer'))['command'])

            for comment in reversed(list(self.search_dict(response, 'commentRenderer'))):
                '''
                result = {'cid': comment['commentId'],
                          'text': ''.join([c['text'] for c in comment['contentText'].get('runs', [])]),
                          'time': comment['publishedTimeText']['runs'][0]['text'],
                          'author': comment.get('authorText', {}).get('simpleText', ''),
                          'channel': comment['authorEndpoint']['browseEndpoint'].get('browseId', ''),
                          'votes': comment.get('voteCount', {}).get('simpleText', '0'),
                          'photo': comment['authorThumbnail']['thumbnails'][-1]['url'],
                          'heart': next(self.search_dict(comment, 'isHearted'), False)}
                '''
                result = {'cid': comment['commentId'],
                          'text': ''.join([c['text'] for c in comment['contentText'].get('runs', [])]),
                          'author': comment.get('authorText', {}).get('simpleText', ''),
                          'votes': comment.get('voteCount', {}).get('simpleText', '0'),
                          'heart': next(self.search_dict(comment, 'isHearted'), False),
                          'paid': None}
                '''
                try:
                    result['time_parsed'] = dateparser.parse(result['time'].split('(')[0].strip()).timestamp()
                except AttributeError:
                    pass
                '''
                paid = (
                    comment.get('paidCommentChipRenderer', {})
                    .get('pdgCommentChipRenderer', {})
                    .get('chipText', {})
                    .get('simpleText')
                )
                if paid:
                    result['paid'] = paid

                yield result
            #time.sleep(sleep)

    @staticmethod
    def regex_search(text, pattern, group=1, default=None):
        match = re.search(pattern, text)
        return match.group(group) if match else default

    @staticmethod
    def search_dict(partial, search_key):
        stack = [partial]
        while stack:
            current_item = stack.pop()
            if isinstance(current_item, dict):
                for key, value in current_item.items():
                    if key == search_key:
                        yield value
                    else:
                        stack.append(value)
            elif isinstance(current_item, list):
                for value in current_item:
                    stack.append(value)

import pandas as pd

downloader = YoutubeCommentDownloader()

import demoji
import re
import string

def clean_text(text):
        # Remove links in tweets
        text = re.sub(r'http\S+', " ", text)
        # Tweets are usually full of emojis. We need to remove them.
        text = demoji.replace(text, repl="")
        return text

def check_comment(cmt):
    t=0
    i=0
    for c in cmt:
        t+=1
        if u'\u0d00' <= c <= u'\u0d7f':
            i+=1
    if t==0 or i/t < 0.4:
        return False
    return True
    
def getComments(videoId):        
    gen = downloader.get_comments(videoId)#, language='ml')
    for cmt in gen:
        yield clean_text(list(cmt.values())[1])

from mlmorph import Analyser
import re

def mlwordPreprocess(w):
    pat = re.compile(r'[^\u0d00-\u0d7f\s]',re.UNICODE)
    return re.sub(pat, " ", w).strip()

def mlmorphStemmer():
    analyser = Analyser()
    def stemit(s):
        #print(s)
        s=mlwordPreprocess(s)
        #print(s)
        res = analyser.analyse(s)
        if not res:
            return s
        res = re.split('<|>', res[0][0])
        i = 0
        rs = ""
        while i < len(res)-1:
            if res[i] == '' or  res[i][0] < u'\u0d00' or res[i][0] > u'\u0d7f':
                i+=2
                continue
            rs += res[i]+" "
            i+=2
        return rs[:-1]
    def stemmer(s):
        return " ".join([stemit(w) for w in s.split()])
    return stemmer

from transformers import AutoModel,AutoTokenizer,PreTrainedModel,PreTrainedTokenizer
import numpy as np
import torch
import pickle 
from xgboost import XGBClassifier

def getpredictor():
    stemmer = mlmorphStemmer()
    model_name="google/muril-large-cased"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
    model.eval()
    clf = None
    with open('xgbm.pkl', 'rb') as f:
        clf = pickle.load(f)
    def predict(text):
        text = stemmer(text)
        tt='''
        tokenized_text = tokenizer.encode(text, add_special_tokens=True, truncation=True)
        max_len = 512
        tokenized_padded_text = np.array([tokenized_text + [0]*(max_len-len(tokenized_text))])
        attention_mask = np.where(tokenized_padded_text != 0, 1, 0)
        input_ids = torch.tensor(tokenized_padded_text)
        attention_mask = torch.tensor(attention_mask)
        last_hidden_states = model(input_ids, attention_mask=attention_mask)
        emb = last_hidden_states[0][:,0,:].numpy()
        '''
        tokenized_text = tokenizer.encode(text, add_special_tokens=True, truncation=True)
        input_ids = torch.tensor(tokenized_text)
        input_ids = input_ids.unsqueeze(0)
        with torch.no_grad():
            last_hidden_states = model(input_ids)
        emb = last_hidden_states[0][:,0,:].numpy()
        return clf.predict(emb)
    return predict

session_state = get_this_session_info()
try:
    tt=session_state.pred
except:
    session_state.pred = getpredictor()
    pass
def main_page():
    st.markdown("# Web app to check sentiment of Malayalam text is Positive, Negative or Neutral 🎈")
    review = st.text_input("Enter Comment...")
    gen_pred = st.button("Predict Sentiment")
    if gen_pred:
        review=review.strip()
        if review!="":
            pred = session_state.pred(review)[0]
            if pred==2:
                st.write("Positive 😀")
            elif pred==1:
                st.write("Negative 😑")
            else:
                st.write("Neutral  😐")

def page2():
    st.markdown("# Analyse youtube video sentiment ❄️")
    videourl = st.text_input("Enter Video link...")
    gen_pred = st.button("Analyse")
    if gen_pred:
        #pred = 1
        videourl=videourl.strip()
        if videourl!="":
            urlid = (videourl.split("/")[-1]).split("=")[-1]
            print(urlid)
            cmts = getComments(urlid)
            cmts = [c for c in cmts if c.strip() and check_comment(c)]
            if len(cmts)==0:
                df = pd.DataFrame({"comment":["Only works on pure malayalam text"],"sentiment":[0]})
            else:
                pred = getpredictor()
                res = []
                for cmt in cmts:
                    it=session_state.pred(cmt)[0]
                    res.append(it)
                df = pd.DataFrame({"comment":cmts,"sentiment":res})    
            #df.columns=["Comment","Sentiment"]
            df["sentiment"] = df["sentiment"].map({2:'Positive 😀', 1:"Negative 😑", 0:"Neutral  😐"})
            #print(df)
            st.dataframe(df, width=600, height=800)

page_names_to_funcs = {
    "Analyse Text": main_page,
    "Analyse Video": page2,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
