# -*- coding: utf-8 -*-
# Streamlit News Recommender — SBERT + (opsional) LR booster, scrape 3 portal ID

import streamlit as st
import re, json, time, random, requests, base64, urllib.parse, math
import pandas as pd
import pytz, feedparser
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from collections import Counter
from github import Github
import streamlit.components.v1 as components

st.set_page_config(page_title="Sistem Rekomendasi Berita", layout="wide")

# --------- state ---------
if 'history' not in st.session_state: st.session_state.history = pd.DataFrame()
if 'current_search_results' not in st.session_state: st.session_state.current_search_results = pd.DataFrame()
if 'show_results' not in st.session_state: st.session_state.show_results = False
if 'current_query' not in st.session_state: st.session_state.current_query = ""
if 'current_recommended_results' not in st.session_state: st.session_state.current_recommended_results = pd.DataFrame()
if 'clicked_urls_in_session' not in st.session_state: st.session_state.clicked_urls_in_session = []

# --------- konstanta ---------
USER_ID = "user_01"
TZ_JKT = pytz.timezone("Asia/Jakarta")
DEFAULT_MIN_SCORE = 0.55
USE_LR_BOOST = True
ALPHA = 0.25
PER_SOURCE_GROUP = True

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0",
]

def make_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "id,en-US;q=0.9,en;q=0.8",
        "Cache-Control": "no-cache", "Pragma": "no-cache",
        "Connection": "keep-alive", "Upgrade-Insecure-Requests": "1",
    })
    return s

def get_source_from_url(url: str) -> str:
    if "detik.com" in url: return "Detik"
    if "cnnindonesia.com" in url: return "CNN"
    if "kompas.com" in url: return "Kompas"
    return "Tidak Diketahui"

@st.cache_resource
def load_sbert():
    try: return SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    except Exception: return SentenceTransformer('paraphrase-MiniLM-L6-v2')
model_sbert = load_sbert()

@st.cache_data
def preprocess_text(t: str) -> str:
    t = (t or "").lower()
    t = re.sub(r'http\S+|www\.\S+','',t)
    t = re.sub(r'\s+',' ',t).strip()
    return t

def _keywords(tokens_min3, hay): return sum(tok in hay for tok in tokens_min3)
def is_relevant_strict(query, title, summary, content, url) -> bool:
    q = preprocess_text(query)
    hay = preprocess_text(" ".join([title or "", summary or "", content or "", url or ""]))
    if len(q.split()) >= 2 and q in hay: return True
    toks = [t for t in re.findall(r"\w+", q) if len(t) >= 3]
    return _keywords(toks, hay) >= 2
def _keywords_ok(title, summary, query) -> bool:
    q = preprocess_text(query)
    toks = [t for t in re.findall(r"\w+", q) if len(t) >= 3]
    hay = preprocess_text((title or "")+" "+(summary or ""))
    return any(tok in hay for tok in toks)

def _has_tz_info(dt_str: str) -> bool:
    if not dt_str: return False
    return bool(re.search(r'(Z|[+\-]\d{2}:\d{2}|[+\-]\d{4})$', dt_str.strip()))

@st.cache_data
def _normalize_to_jakarta(dt_str: str) -> str:
    if not dt_str: return ""
    dt_str = dt_str.strip().replace(" WIB","").replace(",", "")
    try:
        if _has_tz_info(dt_str):
            ts = pd.to_datetime(dt_str, utc=True, errors="coerce")
            if ts is not None and not pd.isna(ts):
                return ts.tz_convert(TZ_JKT).strftime("%Y-%m-%d %H:%M")
        else:
            ts = pd.to_datetime(dt_str, errors="coerce")
            if ts is not None and not pd.isna(ts):
                if ts.tzinfo is None: ts = TZ_JKT.localize(ts)
                else: ts = ts.tz_convert(TZ_JKT)
                return ts.strftime("%Y-%m-%d %H:%M")
    except Exception:
        pass
    return ""

@st.cache_data
def _parse_id_date_text(text: str) -> str:
    if not text: return ""
    t = text.strip()
    m0 = re.search(r'(\d{2})/(\d{2})/(\d{4})[, ]+(\d{2}):(\d{2})', t)
    if m0:
        dd,mm,yyyy,hh,mi = m0.groups()
        return _normalize_to_jakarta(f"{yyyy}-{mm}-{dd} {hh}:{mi}")
    bulan = {"Jan":"01","Feb":"02","Mar":"03","Apr":"04","Mei":"05","Jun":"06","Jul":"07","Agu":"08","Sep":"09","Okt":"10","Nov":"11","Des":"12"}
    m1 = re.search(r"(Senin|Selasa|Rabu|Kamis|Jumat|Sabtu|Minggu)\s*,\s*(\d{1,2})\s+(Jan|Feb|Mar|Apr|Mei|Jun|Jul|Agu|Sep|Okt|Nov|Des)\s+(\d{4})\s+(\d{2}:\d{2})", t, re.I)
    if m1:
        _,dd,mon,yyyy,hhmm = m1.groups()
        mm = bulan.get(mon[:3].title(),"00")
        if mm!="00": return _normalize_to_jakarta(f"{yyyy}-{mm}-{int(dd):02d} {hhmm}")
    m2 = re.search(r"(Senin|Selasa|Rabu|Kamis|Jumat|Sabtu|Minggu)\s*,\s*(\d{2})/(\d{2})/(\d{4})\s+(\d{2}:\d{2})", t, re.I)
    if m2:
        _,dd,mm,yyyy,hhmm = m2.groups()
        return _normalize_to_jakarta(f"{yyyy}-{mm}-{dd} {hhmm}")
    m3 = re.search(r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2})?(?:Z|[+\-]\d{2}:?\d{2})?)", t)
    if m3: return _normalize_to_jakarta(m3.group(1))
    m4 = re.search(r"(\d+)\s+(menit|jam|hari|minggu)\s+yang\s+lalu", t, re.I)
    if m4:
        jumlah = int(m4.group(1)); unit = m4.group(2).lower()
        now = datetime.now(TZ_JKT)
        if unit.startswith("menit"): dt = now - timedelta(minutes=jumlah)
        elif unit.startswith("jam"): dt = now - timedelta(hours=jumlah)
        elif unit.startswith("hari"): dt = now - timedelta(days=jumlah)
        else: dt = now - timedelta(weeks=jumlah)
        return dt.strftime("%Y-%m-%d %H:%M")
    return ""

@st.cache_data
def extract_published_at_from_article_html(art_soup: BeautifulSoup, url: str = "") -> str:
    try:
        for s in art_soup.find_all("script", attrs={"type":"application/ld+json"}):
            raw = (s.string or s.text or "").strip()
            if not raw: continue
            data = json.loads(raw)
            objs = data if isinstance(data, list) else [data]
            for obj in objs:
                if not isinstance(obj, dict): continue
                for k in ("datePublished","dateCreated"):
                    if obj.get(k):
                        t = _normalize_to_jakarta(str(obj[k]))
                        if t: return t
                mep = obj.get("mainEntityOfPage")
                if isinstance(mep, dict) and mep.get("datePublished"):
                    t = _normalize_to_jakarta(str(mep["datePublished"]))
                    if t: return t
    except Exception: pass
    meta = [
        ("property",["article:published_time","og:published_time","og:updated_time"]),
        ("name",["publishdate","pubdate","DC.date.issued","date","content_PublishedDate"]),
        ("itemprop",["datePublished","datecreated"])
    ]
    for attr, names in meta:
        for nm in names:
            tag = art_soup.find("meta", attrs={attr:nm})
            if tag and tag.get("content"):
                t = _normalize_to_jakarta(tag["content"])
                if t: return t
    ttag = art_soup.find("time", attrs={"datetime":True})
    if ttag and ttag.get("datetime"):
        t = _normalize_to_jakarta(ttag["datetime"])
        if t: return t
    for sel in ["div.detail__date","div.read__time","div.date","span.date","span.box__date","div.the_date","p.date","time"]:
        tag = art_soup.select_one(sel)
        if tag:
            t = _parse_id_date_text(tag.get_text(" ", strip=True))
            if t: return t
    if url and "kompas.com" in url:
        m = re.search(r"/(\d{4})/(\d{2})/(\d{2})/(\d{4,8})", url)
        if m:
            y,mo,d,hhmmxx = m.groups()
            hh,mi = hhmmxx[:2], hhmmxx[2:4]
            if hh.isdigit() and mi.isdigit():
                return _normalize_to_jakarta(f"{y}-{mo}-{d} {hh}:{mi}")
    return ""

def fetch_time_and_content(sess, link):
    published_at = ""; content = ""
    try:
        time.sleep(random.uniform(0.35,0.8))
        ar = sess.get(link, timeout=20)
        if ar.status_code == 200:
            art = BeautifulSoup(ar.content, "html.parser")
            published_at = extract_published_at_from_article_html(art, link)
            paras = art.select("article p, div.read__content p, div.detail__body p, .text-article p")
            content = " ".join([p.get_text(" ", strip=True) for p in paras])[:1500]
    except Exception:
        pass
    return published_at, content

# --------- scrapers ---------
@st.cache_data(show_spinner="Mencari berita di Detik...", ttl=300)
def scrape_detik(query, max_articles=12):
    data = []; sess = make_session()
    try:
        search_url = f"https://www.detik.com/search/searchall?query={requests.utils.quote(query)}"
        res = sess.get(search_url, timeout=15)
        if res.status_code == 200:
            soup = BeautifulSoup(res.content, "html.parser")
            items = soup.select("article.list-content__item") or soup.select("li.list__item, div.list__item, div.list-content__item")
            for it in items:
                try:
                    a = it.select_one("h3.media__title a, h2.media__title a, a.media__link, a[href]")
                    if not a or not a.get("href"): continue
                    link = a["href"]; title = a.get_text(strip=True)
                    desc_el = it.select_one(".media__desc, .desc, p")
                    description = desc_el.get_text(strip=True) if desc_el else ""
                    pub, content = fetch_time_and_content(sess, link)
                    if not pub:
                        dt_hint = it.select_one(".media__date, .date")
                        if dt_hint: pub = _parse_id_date_text(dt_hint.get_text(" ", strip=True))
                    if not pub: continue
                    if not is_relevant_strict(query, title, description, content, link): continue
                    data.append({"source":"Detik","title":title,"description":description,
                                 "content":(title+" "+description+" "+content).strip(),
                                 "url":link,"publishedAt":pub})
                    if len(data)>=max_articles: break
                except Exception: continue
    except Exception: pass
    if len(data) < max_articles:
        feeds = ["https://news.detik.com/rss","https://finance.detik.com/rss","https://sport.detik.com/rss",
                 "https://hot.detik.com/rss","https://inet.detik.com/rss","https://health.detik.com/rss",
                 "https://oto.detik.com/rss","https://travel.detik.com/rss","https://food.detik.com/rss","https://20.detik.com/rss"]
        try:
            for fu in feeds:
                if len(data)>=max_articles: break
                feed = feedparser.parse(fu)
                for e in feed.entries:
                    if len(data)>=max_articles: break
                    title = getattr(e,"title",""); link = getattr(e,"link",""); summary = getattr(e,"summary","")
                    if not link or not _keywords_ok(title, summary, query): continue
                    pub = ""
                    if getattr(e,"published_parsed",None):
                        utc_dt = datetime(*e.published_parsed[:6], tzinfo=pytz.UTC)
                        pub = utc_dt.astimezone(TZ_JKT).strftime("%Y-%m-%d %H:%M")
                    real, content = fetch_time_and_content(sess, link)
                    if real: pub = real
                    if not pub: continue
                    if not is_relevant_strict(query, title, summary, content, link): continue
                    data.append({"source":"Detik","title":title,"description":summary,
                                 "content":(title+" "+summary+" "+content).strip(),
                                 "url":link,"publishedAt":pub})
        except Exception: pass
    return pd.DataFrame(data).drop_duplicates(subset=["url"]) if data else pd.DataFrame()

@st.cache_data(show_spinner="Mencari berita di CNN...", ttl=300)
def scrape_cnn_fixed(query, max_results=10):
    urls = [f"https://www.cnnindonesia.com/search?query={requests.utils.quote(query)}",
            "https://www.cnnindonesia.com/nasional/rss","https://www.cnnindonesia.com/internasional/rss",
            "https://www.cnnindonesia.com/ekonomi/rss","https://www.cnnindonesia.com/olahraga/rss",
            "https://www.cnnindonesia.com/gaya-hidup/rss"]
    results = []; sess = make_session()
    for url in urls:
        if len(results)>=max_results: break
        try:
            if "rss" in url:
                feed = feedparser.parse(url)
                for entry in feed.entries:
                    if len(results)>=max_results: break
                    title = getattr(entry,"title",""); link = getattr(entry,"link",""); summary = getattr(entry,"summary","")
                    if not link or not _keywords_ok(title, summary, query): continue
                    pub = ""
                    if getattr(entry,"published_parsed",None):
                        utc_dt = datetime(*entry.published_parsed[:6], tzinfo=pytz.UTC)
                        pub = utc_dt.astimezone(TZ_JKT).strftime("%Y-%m-%d %H:%M")
                    real, content = fetch_time_and_content(sess, link)
                    if real: pub = real
                    if not pub: continue
                    if not is_relevant_strict(query, title, summary, content, link): continue
                    results.append({"source":"CNN","title":title,"description":summary,
                                    "content":(title+" "+summary+" "+content).strip(),
                                    "url":link,"publishedAt":pub})
            else:
                res = sess.get(url, timeout=15)
                if res.status_code==200:
                    soup = BeautifulSoup(res.content,'html.parser')
                    cards = soup.find_all('article','box--card')
                    for a in cards:
                        if len(results)>=max_results: break
                        try:
                            link_tag = a.find('a','box--card__link')
                            if not link_tag: continue
                            link = link_tag['href']
                            ttl_el = link_tag.find('span','box--card__title')
                            title = ttl_el.get_text(strip=True) if ttl_el else ""
                            sum_el = a.find('span','box--card__desc')
                            summary = sum_el.get_text(strip=True) if sum_el else ""
                            pub, content = fetch_time_and_content(sess, link)
                            if not pub: continue
                            if not is_relevant_strict(query, title, summary, content, link): continue
                            results.append({"source":get_source_from_url(link),"title":title,"description":summary,
                                            "content":(title+" "+summary+" "+content).strip(),
                                            "url":link,"publishedAt":pub})
                        except Exception: continue
        except Exception: continue
    return pd.DataFrame(results).drop_duplicates(subset=["url"]) if results else pd.DataFrame()

@st.cache_data(show_spinner="Mencari berita di Kompas...", ttl=300)
def scrape_kompas_fixed(query, max_articles=10):
    data = []; sess = make_session()
    try:
        search_url = f"https://search.kompas.com/search?q={requests.utils.quote(query)}"
        res = sess.get(search_url, timeout=20)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            items = soup.select("div.articleItem") or soup.select("div.article__list, li.article__item, div.iso__item")
            for it in items[:max_articles]:
                try:
                    a = it.select_one("a.article-link, a.article__link, a[href]")
                    h = it.select_one("h2.articleTitle, h3.article__title, h2.article__title")
                    if not a or not h: continue
                    url = a.get("href",""); title = h.get_text(strip=True)
                    if not url or "kompas.com" not in url: continue
                    pub, content = fetch_time_and_content(sess, url)
                    if not pub:
                        ttag = it.select_one(".read__time, .date")
                        if ttag: pub = _parse_id_date_text(ttag.get_text(" ", strip=True))
                    if not pub:
                        m = re.search(r"/(\d{4})/(\d{2})/(\d{2})/(\d{4,8})", url)
                        if m:
                            y,mo,d,hhmm = m.groups()
                            pub = _normalize_to_jakarta(f"{y}-{mo}-{d} {hhmm[:2]}:{hhmm[2:4]}")
                    if not pub: continue
                    if not is_relevant_strict(query, title, "", content, url): continue
                    data.append({"source":"Kompas","title":title,"description":"",
                                 "content":(title+" "+content).strip(),
                                 "url":url,"publishedAt":pub})
                except Exception: continue
    except Exception: pass
    if len(data)<max_articles:
        feeds = ["https://nasional.kompas.com/rss","https://internasional.kompas.com/rss","https://ekonomi.kompas.com/rss",
                 "https://bola.kompas.com/rss","https://tekno.kompas.com/rss","https://sains.kompas.com/rss","https://megapolitan.kompas.com/rss"]
        try:
            for fu in feeds:
                if len(data)>=max_articles: break
                feed = feedparser.parse(fu)
                for e in feed.entries:
                    if len(data)>=max_articles: break
                    title = getattr(e,"title",""); link = getattr(e,"link",""); summary = getattr(e,"summary","")
                    if not link or not _keywords_ok(title, summary, query): continue
                    pub = ""
                    if getattr(e,"published_parsed",None):
                        utc_dt = datetime(*e.published_parsed[:6], tzinfo=pytz.UTC)
                        pub = utc_dt.astimezone(TZ_JKT).strftime("%Y-%m-%d %H:%M")
                    real, content = fetch_time_and_content(sess, link)
                    if real: pub = real
                    if not pub: continue
                    if not is_relevant_strict(query, title, summary, content, link): continue
                    data.append({"source":"Kompas","title":title,"description":summary,
                                 "content":(title+" "+summary+" "+content).strip(),
                                 "url":link,"publishedAt":pub})
        except Exception: pass
    return pd.DataFrame(data).drop_duplicates(subset=["url"]) if data else pd.DataFrame()

@st.cache_data(show_spinner="Menggabungkan hasil...", ttl=300)
def scrape_all_sources(query):
    dfs = []
    for df in [scrape_detik(query), scrape_cnn_fixed(query), scrape_kompas_fixed(query)]:
        if not df.empty: dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# --------- GitHub storage ---------
@st.cache_resource(ttl=60)
def get_github_client():
    try: return Github(st.secrets["github_token"])
    except Exception: return None

@st.cache_data(ttl=60)
def load_history_from_github():
    try:
        g = get_github_client()
        if g is None: return pd.DataFrame()
        repo = g.get_user(st.secrets["repo_owner"]).get_repo(st.secrets["repo_name"])
        contents = repo.get_contents(st.secrets["file_path"])
        data = json.loads(contents.decoded_content.decode('utf-8'))
        if not data: return pd.DataFrame()
        df = pd.DataFrame(data)
        for col in ['user_id','query','click_time','publishedAt']:
            if col not in df.columns: df[col] = None
        return df
    except Exception:
        return pd.DataFrame()

def save_interaction_to_github(user_id, query, all_articles, clicked_urls):
    try:
        g = get_github_client()
        if g is None: return
        repo = g.get_user(st.secrets["repo_owner"]).get_repo(st.secrets["repo_name"])
        try:
            contents = repo.get_contents(st.secrets["file_path"])
            history_str = contents.decoded_content.decode('utf-8')
            history_list = json.loads(history_str)
        except Exception:
            contents = repo.get_contents(st.secrets["file_path"])
            history_list = []
        now = datetime.now(TZ_JKT).strftime("%A, %d %B %Y %H:%M")
        for _, row in all_articles.iterrows():
            history_list.append({
                "user_id": user_id, "query": query,
                "title": str(row.get('title','')), "url": str(row.get('url','')),
                "content": str(row.get('content','')), "source": str(row.get('source','')),
                "click_time": now, "publishedAt": row.get('publishedAt', ""),
                "label": 1 if row.get('url','') in clicked_urls else 0
            })
        updated = json.dumps(history_list, indent=2, ensure_ascii=False)
        repo.update_file(st.secrets["file_path"], f"Update history for {query}", updated, contents.sha)
    except Exception:
        pass

def append_click_to_github(user_id, query, row_dict):
    try:
        g = get_github_client()
        if g is None: return
        repo = g.get_user(st.secrets["repo_owner"]).get_repo(st.secrets["repo_name"])
        contents = repo.get_contents(st.secrets["file_path"])
        history_list = json.loads(contents.decoded_content.decode('utf-8'))
        now = datetime.now(TZ_JKT).strftime("%A, %d %B %Y %H:%M")
        history_list.append({
            "user_id": user_id, "query": query,
            "title": str(row_dict.get('title','')), "url": str(row_dict.get('url','')),
            "content": str(row_dict.get('content','')), "source": str(row_dict.get('source','')),
            "click_time": now, "publishedAt": row_dict.get('publishedAt', ""),
            "label": 1
        })
        updated = json.dumps(history_list, indent=2, ensure_ascii=False)
        repo.update_file(st.secrets["file_path"], f"Append click for {query}", updated, contents.sha)
    except Exception:
        pass

# --------- analitik ---------
def get_recent_queries_by_days(user_id, df, days=3):
    if df.empty or "user_id" not in df.columns or 'click_time' not in df.columns: return {}
    df_user = df[df["user_id"] == user_id].copy()
    if 'publishedAt' not in df_user.columns: df_user['publishedAt'] = df_user['click_time']
    df_user['date_to_process'] = df_user['click_time']
    df_user["timestamp"] = pd.to_datetime(df_user["date_to_process"], format="%A, %d %B %Y %H:%M", errors='coerce')
    df_user["timestamp"].fillna(pd.to_datetime(df_user["date_to_process"], format="%Y-%m-%d %H:%M", errors='coerce'), inplace=True)
    df_user = df_user.dropna(subset=['timestamp']).copy()
    try:
        df_user['timestamp'] = pd.to_datetime(df_user['timestamp']).dt.tz_localize(TZ_JKT, ambiguous='NaT', nonexistent='NaT')
    except Exception:
        return {}
    df_user = df_user.dropna(subset=['timestamp']).copy()
    if df_user.empty: return {}
    now = datetime.now(TZ_JKT); cutoff = now - timedelta(days=days)
    recent_df = df_user[df_user["timestamp"] >= cutoff].copy()
    if recent_df.empty: return {}
    recent_df.loc[:, 'date'] = recent_df['timestamp'].dt.strftime('%d %B %Y')
    grouped = recent_df.groupby('date')['query'].unique().to_dict()
    sorted_dates = sorted(grouped.keys(), key=lambda d: datetime.strptime(d, '%d %B %Y'), reverse=True)
    return {d: grouped[d] for d in sorted_dates}

def get_top_query_last_days(user_id, df, days=3):
    if df.empty or "user_id" not in df.columns: return []
    df_user = df[df["user_id"]==user_id].copy()
    df_user["timestamp"] = pd.to_datetime(df_user["click_time"], format="%A, %d %B %Y %H:%M", errors='coerce')
    df_user["timestamp"].fillna(pd.to_datetime(df_user["click_time"], errors='coerce'), inplace=True)
    df_user = df_user.dropna(subset=["timestamp"])
    now = datetime.now(TZ_JKT); cutoff = now - timedelta(days=days)
    recent = df_user[df_user["timestamp"] >= cutoff]
    if recent.empty: return []
    cnt = Counter(recent["query"])
    return sorted(cnt.items(), key=lambda x: (-x[1], x[0]))

def build_training_data(user_id):
    history_df = load_history_from_github()
    user_data = [h for h in history_df.to_dict('records')
                 if h.get("user_id")==user_id and "label" in h and h.get("title") and h.get("content")]
    df = pd.DataFrame(user_data)
    if df.empty or df["label"].nunique()<2: return pd.DataFrame()
    train = []; seen=set()
    for _, row in df.iterrows():
        text = preprocess_text(str(row.get("title",""))+" "+str(row.get("content","")))
        label = int(row.get("label",0))
        if text and text not in seen:
            train.append({"text":text,"label":label}); seen.add(text)
    return pd.DataFrame(train)

@st.cache_resource(show_spinner="Melatih model rekomendasi...")
def train_model(df_train):
    X = model_sbert.encode(df_train["text"].tolist())
    y = df_train["label"].tolist()
    if len(set(y))<2:
        st.warning("⚠️ Gagal melatih model: hanya ada satu jenis label.")
        return None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    st.sidebar.markdown("**Evaluasi Model**")
    st.sidebar.write(f"Akurasi: {accuracy_score(y_te, y_pred):.2f}")
    st.sidebar.write(f"Presisi: {precision_score(y_te, y_pred):.2f}")
    st.sidebar.write(f"Recall: {recall_score(y_te, y_pred):.2f}")
    st.sidebar.write(f"Skor F1: {f1_score(y_te, y_pred):.2f}")
    return clf

# --------- ranking ---------
def recommend(df, query, clf, n_per_source=3, min_score=DEFAULT_MIN_SCORE,
              use_lr_boost=USE_LR_BOOST, alpha=ALPHA, per_source_group=PER_SOURCE_GROUP):
    if df.empty: return pd.DataFrame()
    df = df.copy().drop_duplicates(subset=['url'])
    df["processed"] = df.apply(lambda r: preprocess_text(f"{r.get('title','')} {r.get('description','')} {r.get('content','')}"), axis=1)
    art_vecs = model_sbert.encode(df["processed"].tolist())
    df['publishedAt_dt'] = pd.to_datetime(df['publishedAt'], errors='coerce'); df = df.dropna(subset=['publishedAt_dt'])
    if df.empty: return pd.DataFrame()
    q_vec = model_sbert.encode([preprocess_text(query)])
    sims = cosine_similarity(q_vec, art_vecs[:len(df)])[0]
    s_min, s_max = float(sims.min()), float(sims.max())
    sbert_score = (sims - s_min) / (s_max - s_min) if s_max>s_min else sims
    df["sbert_score"] = sbert_score
    if clf is not None and use_lr_boost:
        lr_score = clf.predict_proba(art_vecs[:len(df)])[:,1]
        bonus = df["title"].apply(lambda t: 0.05 if (t and query.lower() in t.lower()) else 0).values
        final = ((1-alpha)*df["sbert_score"].values) + (alpha*lr_score) + bonus
    else:
        final = df["sbert_score"].values
    df["final_score"] = final.clip(0,1)
    filtered = df[df['final_score']>=min_score].copy()
    if filtered.empty: filtered = df.copy()
    def _top_n(x):
        return x.sort_values(["publishedAt_dt","final_score"], ascending=[False,False]).head(n_per_source)
    got = filtered.groupby("source", group_keys=False).apply(_top_n, include_groups=False) if per_source_group \
          else filtered.sort_values(["publishedAt_dt","final_score"], ascending=[False,False]).head(3*n_per_source)
    return got.sort_values(["publishedAt_dt","final_score"], ascending=[False,False]).reset_index(drop=True)

def format_display_time(s: str) -> str:
    try: return datetime.strptime(s, "%Y-%m-%d %H:%M").strftime("%Y-%m-%d %H:%M")
    except Exception: return "—"

# --------- link logger (button) ---------
def _enc(s): 
    try: return base64.urlsafe_b64encode((s or "").encode()).decode()
    except Exception: return ""
def _dec(s):
    try: return base64.urlsafe_b64decode((s or "").encode()).decode()
    except Exception: return ""

def make_logged_link(row, query, label="Baca selengkapnya"):
    qs = urllib.parse.urlencode({
        "log": _enc(row["url"]), "q": _enc(query),
        "t": _enc(row.get("title","")), "s": _enc(row.get("source","")),
        "p": _enc(row.get("publishedAt","")), "c": _enc(row.get("content",""))
    })
    return f'<a class="btn-read" href="?{qs}" target="_self">{label}</a>'

def intercept_query_params_and_log():
    try:
        params = st.query_params  # Streamlit baru
    except Exception:
        params = st.experimental_get_query_params()
    if "log" in params:
        raw = lambda k: params.get(k, [""])[0] if isinstance(params.get(k, [""]), list) else params.get(k, "")
        url = _dec(raw("log")); qx = _dec(raw("q"))
        row = {"url":url, "title":_dec(raw("t")), "source":_dec(raw("s")),
               "publishedAt":_dec(raw("p")), "content":_dec(raw("c"))}
        if url and (url not in st.session_state.clicked_urls_in_session):
            st.session_state.clicked_urls_in_session.append(url)
            append_click_to_github(USER_ID, qx, row)
        components.html(f"""
            <script>
              try {{
                window.open("{url}", "_blank");
                const base = window.location.href.split("?")[0];
                window.history.replaceState({{}}, "", base);
              }} catch(e) {{}}
            </script>
        """, height=0)
        st.stop()

# --------- UI ---------
st.markdown("""
<style>
.btn-read{display:inline-block;background:#2563eb;color:#fff !important;padding:8px 14px;border-radius:8px;text-decoration:none;font-weight:600}
.btn-read:hover{opacity:.9}
hr{border:0;border-top:1px solid rgba(255,255,255,.1)}
</style>""", unsafe_allow_html=True)

def main():
    intercept_query_params_and_log()

    st.title("📰 Rekomendasi Berita")
    st.markdown("Aplikasi merekomendasikan berita dari Detik, CNN, dan Kompas berdasarkan riwayat & topik. Waktu publikasi diambil langsung dari halaman artikel.")

    if st.sidebar.button("Bersihkan Cache & Muat Ulang"):
        st.cache_data.clear(); st.cache_resource.clear()
        st.success("Cache dibersihkan. Memuat ulang…"); time.sleep(1); st.rerun()

    if st.session_state.history.empty:
        st.session_state.history = load_history_from_github()

    st.sidebar.header("Model Personalisasi")
    df_train = build_training_data(USER_ID)
    clf = None
    if not df_train.empty and df_train['label'].nunique()>1:
        st.sidebar.success("Model berhasil dilatih.")
        clf = train_model(df_train)
    else:
        st.sidebar.info("Model belum bisa dilatih karena riwayat tidak mencukupi.")

    st.header("📚 Pencarian Berita per Tanggal")
    grouped = get_recent_queries_by_days(USER_ID, st.session_state.history, days=3)
    if grouped:
        for date, queries in grouped.items():
            st.subheader(f"Tanggal {date}")
            for q in sorted(set(queries)):
                with st.expander(f"- {q}", expanded=False):
                    with st.spinner('Mengambil berita terbaru...'):
                        df_latest = scrape_all_sources(q)
                    if df_latest.empty:
                        st.info("❗ Tidak ditemukan berita terbaru untuk topik ini.")
                        continue
                    results_latest = recommend(df_latest, q, clf, n_per_source=3)
                    if results_latest.empty:
                        st.info("❗ Tidak ada hasil relevan dari portal untuk topik ini.")
                        continue
                    for _, row in results_latest.iterrows():
                        src = get_source_from_url(row['url'])
                        st.markdown(f"**[{src}]** {row['title']}")
                        st.markdown(f"[{row['url']}]({row['url']})")
                        st.write(f"Waktu Publikasi: *{format_display_time(row.get('publishedAt',''))}*")
                        skor = float(row.get('final_score', row.get('sbert_score', 0.0)))
                        st.write(f"Skor: `{skor:.2f}`")
                        st.markdown("---")
    else:
        st.info("Belum ada riwayat pencarian pada 3 hari terakhir.")

    st.markdown("---")

    st.header("🔥 Rekomendasi Berita Hari Ini")
    trends = get_top_query_last_days(USER_ID, st.session_state.history, days=3)
    if trends:
        q, _ = trends[0]
        with st.spinner('Mencari berita...'):
            df_news = scrape_all_sources(q)
        if df_news.empty:
            st.info("❗ Tidak ditemukan berita.")
        else:
            results = recommend(df_news, q, clf, n_per_source=1)
            if results.empty:
                st.info("❗ Tidak ada hasil relevan.")
            else:
                for _, row in results.iterrows():
                    src = get_source_from_url(row['url'])
                    st.markdown(f"**[{src}]** {row['title']}")
                    st.markdown(f"[{row['url']}]({row['url']})")
                    st.write(f"Waktu: *{format_display_time(row.get('publishedAt',''))}*")
                    skor = float(row.get('final_score', row.get('sbert_score', 0.0)))
                    st.write(f"Skor: `{skor:.2f}`")
                    st.markdown(make_logged_link(row, q), unsafe_allow_html=True)
                    st.markdown("---")
    else:
        st.info("🔥 Tidak ada topik yang sering dicari dalam 3 hari terakhir.")

    st.markdown("---")

    st.header("🔍 Pencarian Berita")
    search_query = st.text_input("Ketik topik berita yang ingin Anda cari:", key="search_input")
    if st.button("Cari Berita"):
        if search_query:
            if st.session_state.current_query:
                save_interaction_to_github(
                    USER_ID, st.session_state.current_query,
                    st.session_state.current_recommended_results,
                    st.session_state.clicked_urls_in_session
                )
                st.cache_data.clear()
                st.session_state.history = load_history_from_github()
            with st.spinner('Mengambil berita dan merekomendasikan...'):
                st.session_state.current_search_results = scrape_all_sources(search_query)
                st.session_state.current_recommended_results = recommend(
                    st.session_state.current_search_results, search_query, clf, n_per_source=3
                )
            st.session_state.show_results = True
            st.session_state.current_query = search_query
            st.session_state.clicked_urls_in_session = []
            st.rerun()
        else:
            st.warning("Mohon masukkan topik pencarian.")

    if st.session_state.show_results:
        st.subheader(f"📌 Hasil untuk '{st.session_state.current_query}'")
        if st.session_state.current_recommended_results.empty:
            st.warning("❗ Tidak ada hasil yang relevan. Coba kata kunci lain.")
        else:
            for _, row in st.session_state.current_recommended_results.iterrows():
                src = get_source_from_url(row['url'])
                st.markdown(f"**[{src}]** {row['title']}")
                st.markdown(f"[{row['url']}]({row['url']})")
                st.write(f"Waktu: *{format_display_time(row.get('publishedAt',''))}*")
                skor = float(row.get('final_score', row.get('sbert_score', 0.0)))
                st.write(f"Skor: `{skor:.2f}`")
                st.markdown(make_logged_link(row, st.session_state.current_query), unsafe_allow_html=True)
                st.markdown("---")
        if st.session_state.current_query:
            st.info(f"Klik tercatat: {len(st.session_state.clicked_urls_in_session)}. Riwayat akan disimpan saat Anda memulai pencarian baru.")

if __name__ == "__main__":
    main()
