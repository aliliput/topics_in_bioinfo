# coding: utf-8
import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize
from collections import Counter
import pandas as pd
import numpy as np
from pandas import Series

#元データを取得
def get_soup(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'}
    r = requests.get(url, headers=headers).content
    soup = BeautifulSoup(r, "lxml")
    return soup

def get_links(url, link_tags):
    links = []
    for a in link_tags:
        link = url + a.attrs['href']
        links.append(link)
    return links

#文章を分解して品詞つき単語リストを作る
def get_tagged_list(text_list):
    tagged_list = []
    for s in text_list:
        text = word_tokenize(s)
        tagged = nltk.pos_tag(text)
        tagged_list.append(tagged)
    return tagged_list

#品詞付き単語リストから複合名詞を抽出
def get_compound_nouns(tagged_list):
    nns = ["NN", "NNS", "NNP", "NNPS"]
    words_list = []
    word = ""

    for tl in tagged_list:
        for w in tl:
            if w[1] in nns:
                word = word + w[0] + " "
            elif word is not "":
                word = word.strip()
                words_list.append(word)
                word = ""
    return words_list

#複合名詞のリストをSeriesに変換し単語を集計
def count_words(words_list):
    c = Series(words_list)
    words_ser = c.value_counts(sort=True)
    return words_ser
    #words_ser[words_ser>4]

#文章のデータをSeriesにする
def make_ser(contents):
    tagged_list = get_tagged_list(contents)
    words_list = get_compound_nouns(tagged_list)
    words = count_words(words_list)
    return words

#新旧のSeriesをDataFrameに結合し増加率を算出
def get_inc_rate(past, latest, th):
    dframe = pd.concat([past, latest], keys=['Past', 'Latest'])
    dframe = dframe.unstack().T
    #新しいデータに閾値以下の出現数しかない単語を削除
    dframe = dframe[dframe.Latest>th].fillna(0).sort_values(by='Latest', ascending=False)
    #増加率を算出して追加。新しく出現した単語の場合は100％とする
    dframe['Rate'] = dframe['Latest']/dframe['Past']
    dframe = dframe[dframe.Rate>0].replace([np.inf, -np.inf], 100).sort_values(by='Rate', ascending=False)
    return dframe

#Oxford academicの新旧データを取る
def get_links_oa(soup):
    #各論文のアブスト掲載ページへのURLを取得
    link_tags = soup.find_all(class_="viewArticleLink")
    links = []
    for a in link_tags:
        link = "https://academic.oup.com" + a.attrs['href']
        links.append(link)
    return links

def get_contents_oa(links):
    contents = []
    for l in links:
        soup = get_soup(l)
        title_txt = soup.find(class_="wi-article-title").get_text()
        title_txt = title_txt.rstrip()
        abst = soup.find(class_="abstract")
        if abst is not None:
            if abst.strong is not None:
                remove = abst.strong.extract()
            abst_data = abst.find_all("p")
            abst_txt = abst_data[0].get_text() + abst_data[1].get_text()
            contents.append(title_txt)
            contents.append(title_txt)
            contents.append(abst_txt)
    return contents

#新しい方のOAのコンテンツを取得
soup = get_soup("https://academic.oup.com/bioinformatics/issue/33/12")

links = get_links_oa(soup)

oa_contents_latest = get_contents_oa(links)

#古い方のOAのコンテンツを取得
soup = get_soup("https://academic.oup.com/bioinformatics/issue/32/12")

links = get_links_oa(soup)
oa_contents_past = get_contents_oa(links)

def get_contents_bmc(links):
    contents_bmc = []
    for l in links:
        soup = get_soup(l)
        title_txt = soup.find(class_="ArticleTitle").get_text()
        title_txt = title_txt.rstrip()
        title_txt = re.sub('\.\S', ". ", title_txt)
        abst = soup.find(class_="js-CollapseSection")
        abst_txt = ""
        if abst is not None:
            abst = abst.find_all("p")
            for a in abst:
                abst_txt = abst_txt + a.get_text()
                abst_txt = re.sub('\.\S', ". ", abst_txt)

        contents_bmc.append(title_txt)
        contents_bmc.append(title_txt)
        contents_bmc.append(abst_txt)
    return contents_bmc

def get_links_bmc(soup):
    link_tags = soup.find_all(class_="fulltexttitle")
    links = []
    for a in link_tags:
        link = "https://bmcbioinformatics.biomedcentral.com" + a.attrs['href']
        links.append(link)
    return links

#新しい方のBMCのコンテンツを取得
soup = get_soup("https://bmcbioinformatics.biomedcentral.com/articles?searchType=journalSearch&sort=PubDate&page=1")

links = get_links_bmc(soup)

latest_contents_bmc = get_contents_bmc(links)

#古い方のBMCのコンテンツを取得
soup = get_soup("https://bmcbioinformatics.biomedcentral.com/articles?searchType=journalSearch&sort=PubDate&page=25")

links = get_links_bmc(soup)
past_contents_bmc = get_contents_bmc(links)

def get_contents_ps(links):
    contents_ps = []
    for l in links:
        soup = get_soup(l)
        title_txt = soup.find(class_="article-header__title").get_text()
        abst_txt = ""
        if soup.find(id="abstract") is not None:
            abst_txt = soup.find(id="abstract").p.get_text()

        contents_ps.append(title_txt)
        contents_ps.append(title_txt)
        contents_ps.append(abst_txt)
    return contents_ps

def get_ps_links(soup):
    link_tags = soup.find(id="issueTocGroups").find_all(class_ = "citation")
    for tag in link_tags:
        if tag.find("ul", class_="productMenu") is not None:
            tag.find("ul", class_="productMenu").decompose()
    links = []
    for l in link_tags:
        link = "http://onlinelibrary.wiley.com" + l.a.attrs['href']
        links.append(link)
    return links

#新しい方のProtein Scienceのデータを取得
soup = get_soup("http://onlinelibrary.wiley.com/doi/10.1002/pro.v26.6/issuetoc")

links1 = get_ps_links(soup)

#Protein Scienceは1volごとの量が少ないので2冊まとめる
soup = get_soup("http://onlinelibrary.wiley.com/doi/10.1002/pro.v26.5/issuetoc")

links2 = get_ps_links(soup)

links = links1 + links2

latest_contents_ps = get_contents_ps(links)

#古い方のPSのデータを取得
soup = get_soup("http://onlinelibrary.wiley.com/doi/10.1002/pro.v25.6/issuetoc")

links1 = get_ps_links(soup)

soup = get_soup("http://onlinelibrary.wiley.com/doi/10.1002/pro.v25.5/issuetoc")

link2 = get_ps_links(soup)

links = links1 + link2

past_contents_ps = get_contents_ps(links)

def get_contents_protein(links):
    contents_prot = []
    for l in links:
        soup = get_soup(l)
        title_txt = soup.find(class_="article-header__title").get_text()
        abst_txt = soup.find(id="abstract")
        if abst_txt is not None:
            abst_txt = abst_txt.p.get_text()
            abst_txt = abst_txt.replace("©", "").replace("Wiley Periodicals, Inc.", "")

            contents_prot.append(title_txt)
            contents_prot.append(title_txt)
            contents_prot.append(abst_txt)
    return contents_prot

#新しい方のProteinのデータを取得。Protein誌のリンク仕様はProtein Scienceと同じなのでリンク取得はそっちを流用
soup = get_soup("http://onlinelibrary.wiley.com/doi/10.1002/prot.v85.6/issuetoc")

links1 = get_ps_links(soup)

soup = get_soup("http://onlinelibrary.wiley.com/doi/10.1002/prot.v85.5/issuetoc")

links2 = get_ps_links(soup)

links = links1 + links2

latest_contents_protein = get_contents_protein(links)

#古い方のProteinのデータを取得
soup = get_soup("http://onlinelibrary.wiley.com/doi/10.1002/prot.v84.6/issuetoc")
links1 = get_ps_links(soup)

soup = get_soup("http://onlinelibrary.wiley.com/doi/10.1002/prot.v84.5/issuetoc")
links2 = get_ps_links(soup)

links = links1 + links2
past_contents_protein = get_contents_protein(links)

#全てのデータを合わせる
latest = oa_contents_latest + latest_contents_bmc + latest_contents_ps + latest_contents_protein
past = oa_contents_past + past_contents_bmc + past_contents_ps + past_contents_protein

latest_ser = make_ser(latest)

past_ser = make_ser(past)

words_dframe = get_inc_rate(past_ser, latest_ser, 8)

words_dframe[words_dframe.Rate>1]
