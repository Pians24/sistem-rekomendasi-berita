# -*- coding: utf-8 -*-
import streamlit as st
import re, json, time, random, requests, urllib.parse, hashlib
import pandas as pd
import pytz
import feedparser
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from github import Github
from github.GithubException import GithubException
import streamlit.components.v1 as components

# ========================= KONFIG & STATE =========================
st.set_page_config(page_title="Sistem Rekomendasi Berita", layout="wide")

USER_ID = "user_01"
TZ_JKT = pytz.timezone("Asia/Jakarta")

DEFAULT_MIN_SCORE = 0.55
USE_LR_BOOST = True
ALPHA = 0.25
PER_SOURCE_GROUP = True

S = st.session_state
for k, v in {
    "history": pd.DataFrame(),
    "current_search_results": pd.DataFrame(),
    "show_results": False,
    "current_query": "",
    "current_recommended_results": pd.DataFrame(),
    "clicked_by_query": {},               # klik per kueri -> {query: set(url, ...)}
    "trending_results_df": pd.DataFrame(),
    "trending_query": "",
    "topic_change_counter": 0,
    "per_tanggal_done_counter": 0,
}.items():
    if k not in S:
        S[k] = v

# ======== SKOR CHIP (SANGAT SEDERHANA, INLINE STYLE) ========
def render_score_badge(score: float, label: str = "Skor"):
    """Contoh: Skor: [ 0.51 ] kecil, hijau, kapsul gelap."""
    try:
        val = float(score)
    except Exception:
        val = 0.0
    html = (
        f"{label}: "
        f"<span style=\"display:inline-block;padding:2px 8px;border-radius:6px;"
        f"background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.10);\">"
        f"<span style=\"font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"
        f"'Liberation Mono','Courier New',monospace;font-weight:700;letter-spacing:.2px;"
        f"color:#2ecc71;\">{val:.2f}</span></span>"
    )
    st.markdown(html, unsafe_allow_html=True)

# ========================= HTTP SESSION =========================
UA = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0",
]
def make_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": random.choice(UA),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "id,en-US;q=0.9,en;q=0.8",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    })
    return s

def get_source_from_url(url):
    if "detik.com" in url: return "Detik"
    if "cnnindonesia.com" in url: return "CNN"
    if "kompas.com" in url: return "Kompas"
    return "Tidak Diketahui"

# ========================= MODEL =========================
@st.cache_resource
def load_model():
    try:
        return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    except Exception:
        return SentenceTransformer("paraphrase-MiniLM-L6-v2")
model_sbert = load_model()

@st.cache_data
def preprocess_text(t):
    t = (t or "").lower()
    t = re.sub(r"http\S+|www\.\S+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# ========================= RELEVANSI =========================
def _keywords(tokens_min3, hay):
    return sum(tok in hay for tok in tokens_min3)

def is_relevant_strict(query, title, summary, content, url):
    q = preprocess_text(query)
    hay = preprocess_text(" ".join([title or "", summary or "", content or "", url or ""]))
    if len(q.split()) >= 2 and q in hay:
        return True
    toks = [t for t in re.findall(r"\w+", q) if len(t) >= 3]
    if not toks:
        return False
    need = 1 if len(toks) <= 2 else 2
    return _keywords(toks, hay) >= need

def _keywords_ok(title, summary, query):
    q = preprocess_text(query)
    toks = [t for t in re.findall(r"\w+", q) if len(t) >= 3]
    hay = preprocess_text((title or "") + " " + (summary or ""))
    return any(tok in hay for tok in toks)

# ========================= UTIL JUDUL =========================
DETIK_SUFFIXES = (" - detikNews", " - detikcom", " | detikcom", " | detikNews")
def _clean_title(t: str) -> str:
    t = (t or "").strip()
    for suf in DETIK_SUFFIXES:
        if t.endswith(suf):
            t = t[: -len(suf)]
    return t

def slug_to_title(url: str) -> str:
    try:
        path = urllib.parse.urlparse(url).path or ""
        parts = [p for p in path.split("/") if p]
        seg = parts[-1] if parts else ""
        for i, p in enumerate(parts):
            if p.startswith("d-") and i + 1 < len(parts):
                seg = parts[i + 1]; break
        seg = urllib.parse.unquote(seg)
        seg = re.sub(r"^\d+[-_]*", "", seg)
        seg = seg.replace("-", " ").replace("_", " ").strip()
        return seg.title() if seg else ""
    except Exception:
        return ""

def extract_title_from_article_html(art_soup: BeautifulSoup) -> str:
    m = art_soup.find("meta", attrs={"property":"og:title"})
    if m and m.get("content"): return _clean_title(m["content"])
    m = art_soup.find("meta", attrs={"name":"twitter:title"})
    if m and m.get("content"): return _clean_title(m["content"])
    if art_soup.title and art_soup.title.string:
        return _clean_title(art_soup.title.string)
    for sel in ["h1", "h1.detail__title", "h1.read__title", ".title", ".detail__title"]:
        h = art_soup.select_one(sel)
        if h:
            tx = h.get_text(" ", strip=True)
            if tx: return _clean_title(tx)
    return ""

# ========================= WAKTU ARTIKEL =========================
def _has_tz_info(s):
    if not s: return False
    return bool(re.search(r"(Z|[+\-]\d{2}:\d{2}|[+\-]\d{4})$", s.strip()))

@st.cache_data
def _normalize_to_jakarta(dt_str):
    if not dt_str: return ""
    dt_str = dt_str.strip().replace(" WIB", "").replace(",", "")
    try:
        if _has_tz_info(dt_str):
            ts = pd.to_datetime(dt_str, utc=True, errors="coerce")
            if ts is not None and not pd.isna(ts):
                ts = ts.tz_convert(TZ_JKT)
                return ts.strftime("%Y-%m-%d %H:%M")
        else:
            ts = pd.to_datetime(dt_str, errors="coerce")
            if ts is not None and not pd.isna(ts):
                if ts.tzinfo is None:
                    ts = TZ_JKT.localize(ts)
                else:
                    ts = ts.tz_convert(TZ_JKT)
                return ts.strftime("%Y-%m-%d %H:%M")
    except Exception:
        pass
    return ""

@st.cache_data
def _parse_id_date_text(text):
    if not text: return ""
    t = text.strip()
    m0 = re.search(r"(\d{2})/(\d{2})/(\d{4})[, ]+(\d{2}):(\d{2})", t)
    if m0:
        dd, mm, yyyy, hh, mi = m0.groups()
        return _normalize_to_jakarta(f"{yyyy}-{mm}-{dd} {hh}:{mi}")
    bulan_map = {"Jan":"01","Feb":"02","Mar":"03","Apr":"04","Mei":"05","Jun":"06","Jul":"07","Agu":"08","Sep":"09","Okt":"10","Nov":"11","Des":"12"}
    m1 = re.search(r"(Senin|Selasa|Rabu|Kamis|Jumat|Sabtu|Minggu)\s*,\s*(\d{1,2})\s+(Jan|Feb|Mar|Apr|Mei|Jun|Jul|Agu|Sep|Okt|Nov|Des)\s+(\d{4})\s+(\d{2}:\d{2})", t, flags=re.IGNORECASE)
    if m1:
        _, dd, mon, yyyy, hhmm = m1.groups()
        mm = bulan_map.get(mon[:3].title(), "00")
        if mm != "00": return _normalize_to_jakarta(f"{yyyy}-{mm}-{int(dd):02d} {hhmm}")
    m2 = re.search(r"(Senin|Selasa|Rabu|Kamis|Jumat|Sabtu|Minggu)\s*,\s*(\d{2})/(\d{2})/(\d{4})\s+(\d{2}:\d{2})", t, flags=re.IGNORECASE)
    if m2:
        _, dd, mm, yyyy, hhmm = m2.groups()
        return _normalize_to_jakarta(f"{yyyy}-{mm}-{dd} {hhmm}")
    m3 = re.search(r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2})?(?:Z|[+\-]\d{2}:?\d{2})?)", t)
    if m3: return _normalize_to_jakarta(m3.group(1))
    m4 = re.search(r"(\d+)\s+(menit|jam|hari|minggu)\s+yang\s+lalu", t, flags=re.IGNORECASE)
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
def extract_published_at_from_article_html(art_soup, url=""):
    try:
        for s in art_soup.find_all("script", attrs={"type":"application/ld+json"}):
            raw = (s.string or s.text or "").strip()
            if not raw:
                continue
            data = json.loads(raw)
            candidates = data if isinstance(data, list) else [data]
            for obj in candidates:
                if not isinstance(obj, dict):
                    continue
                for key in ("datePublished","dateCreated"):
                    if obj.get(key):
                        t = _normalize_to_jakarta(str(obj[key]))
                        if t: return t
                mep = obj.get("mainEntityOfPage")
                if isinstance(mep, dict) and mep.get("datePublished"):
                    t = _normalize_to_jakarta(str(mep["datePublished"]))
                    if t: return t
    except Exception:
        pass
    meta_candidates = [
        ("property", ["article:published_time","og:published_time","og:updated_time"]),
        ("name", ["publishdate","pubdate","DC.date.issued","date","content_PublishedDate"]),
        ("itemprop", ["datePublished","datecreated"])
    ]
    for attr, names in meta_candidates:
        for nm in names:
            tag = art_soup.find("meta", attrs={attr:nm})
            if tag and tag.get("content"):
                t = _normalize_to_jakarta(tag["content"])
                if t: return t
    ttag = art_soup.find("time", attrs={"datetime": True})
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
            y, mo, d, hhmmxx = m.groups()
            hh = hhmmxx[:2]; mi = hhmmxx[2:4]
            if hh.isdigit() and mi.isdigit():
                return _normalize_to_jakarta(f"{y}-{mo}-{d} {hh}:{mi}")
    return ""

def fetch_time_content_title(sess, link):
    pub, content, title_html = "", "", ""
    try:
        time.sleep(random.uniform(0.3, 0.7))
        ar = sess.get(link, timeout=20)
        if ar.status_code == 200:
            soup = BeautifulSoup(ar.content, "html.parser")
            pub = extract_published_at_from_article_html(soup, link)
            title_html = extract_title_from_article_html(soup)
            paras = soup.select("article p, div.read__content p, div.detail__body p, .text-article p")
            content = " ".join([p.get_text(" ", strip=True) for p in paras])[:1500]
    except Exception:
        pass
    return pub, content, title_html

def format_display_time(s):
    try:
        dt_obj = datetime.strptime(s, "%Y-%m-%d %H:%M")
        return dt_obj.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "—"

# ========================= SCRAPERS =========================
@st.cache_data(show_spinner="Mencari berita di Detik...", ttl=300)
def scrape_detik(query, max_articles=15):
    data, sess = [], make_session()
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
                    pub, content, title_html = fetch_time_content_title(sess, link)
                    if not pub:
                        dt_hint = it.select_one(".media__date, .date")
                        if dt_hint: pub = _parse_id_date_text(dt_hint.get_text(" ", strip=True))
                    if not pub: continue
                    if not title or len(title) < 3:
                        title = title_html or slug_to_title(link)
                    if not is_relevant_strict(query, title, description, content, link): continue
                    data.append({
                        "source":"Detik","title":title,"description":description,
                        "content": (title+" "+description+" "+content).strip(),
                        "url":link,"publishedAt":pub
                    })
                    if len(data) >= max_articles: break
                except Exception:
                    continue
    except Exception:
        pass
    if len(data) < max_articles:
        feeds = [
            "https://news.detik.com/rss","https://finance.detik.com/rss","https://sport.detik.com/rss",
            "https://hot.detik.com/rss","https://inet.detik.com/rss","https://health.detik.com/rss",
            "https://oto.detik.com/rss","https://travel.detik.com/rss","https://food.detik.com/rss",
            "https://20.detik.com/rss",
        ]
        try:
            for fu in feeds:
                if len(data) >= max_articles: break
                feed = feedparser.parse(fu)
                for e in feed.entries:
                    if len(data) >= max_articles: break
                    title = getattr(e,"title",""); link = getattr(e,"link",""); summary = getattr(e,"summary","")
                    if not link or not _keywords_ok(title, summary, query): continue
                    pub = ""
                    if getattr(e,"published_parsed",None):
                        utc_dt = datetime(*e.published_parsed[:6], tzinfo=pytz.UTC)
                        pub = utc_dt.astimezone(TZ_JKT).strftime("%Y-%m-%d %H:%M")
                    real, content, title_html = fetch_time_content_title(sess, link)
                    if real: pub = real
                    if not pub: continue
                    if not title or len(title) < 3:
                        title = title_html or slug_to_title(link)
                    if not is_relevant_strict(query, title, summary, content, link): continue
                    data.append({
                        "source":"Detik","title":title,"description":summary,
                        "content": (title+" "+summary+" "+content).strip(),
                        "url":link,"publishedAt":pub
                    })
        except Exception:
            pass
    return pd.DataFrame(data).drop_duplicates(subset=["url"]) if data else pd.DataFrame()

@st.cache_data(show_spinner="Mencari berita di CNN...", ttl=300)
def scrape_cnn_fixed(query, max_results=12):
    urls = [
        f"https://www.cnnindonesia.com/search?query={requests.utils.quote(query)}",
        "https://www.cnnindonesia.com/nasional/rss","https://www.cnnindonesia.com/internasional/rss",
        "https://www.cnnindonesia.com/ekonomi/rss","https://www.cnnindonesia.com/olahraga/rss",
        "https://www.cnnindonesia.com/gaya-hidup/rss",
    ]
    results, sess = [], make_session()
    for url in urls:
        if len(results) >= max_results: break
        try:
            if "rss" in url:
                feed = feedparser.parse(url)
                for entry in feed.entries:
                    if len(results) >= max_results: break
                    title = getattr(entry,"title",""); link = getattr(entry,"link",""); summary = getattr(entry,"summary","")
                    if not link or not _keywords_ok(title, summary, query): continue
                    pub = ""
                    if getattr(entry,"published_parsed",None):
                        utc_dt = datetime(*entry.published_parsed[:6], tzinfo=pytz.UTC)
                        pub = utc_dt.astimezone(TZ_JKT).strftime("%Y-%m-%d %H:%M")
                    real, content, title_html = fetch_time_content_title(sess, link)
                    if real: pub = real
                    if not pub: continue
                    if not title or len(title) < 3:
                        title = title_html or slug_to_title(link)
                    if not is_relevant_strict(query, title, summary, content, link): continue
                    results.append({
                        "source":"CNN","title":title,"description":summary,
                        "content": (title+" "+summary+" "+content).strip(),
                        "url":link,"publishedAt":pub
                    })
            else:
                res = sess.get(url, timeout=15)
                if res.status_code == 200:
                    soup = BeautifulSoup(res.content, "html.parser")
                    articles = soup.find_all("article", class_="box--card")
                    for art in articles:
                        if len(results) >= max_results: break
                        try:
                            link_tag = art.find("a", class_="box--card__link")
                            if not link_tag: continue
                            link = link_tag["href"]
                            title_el = link_tag.find("span", class_="box--card__title")
                            desc_el = art.find("span", class_="box--card__desc")
                            title = title_el.get_text(strip=True) if title_el else ""
                            summary = desc_el.get_text(strip=True) if desc_el else ""
                            pub, content, title_html = fetch_time_content_title(sess, link)
                            if not pub: continue
                            if not title or len(title) < 3:
                                title = title_html or slug_to_title(link)
                            if not is_relevant_strict(query, title, summary, content, link): continue
                            results.append({
                                "source": get_source_from_url(link),"title":title,"description":summary,
                                "content": (title+" "+summary+" "+content).strip(),
                                "url":link,"publishedAt":pub
                            })
                        except Exception:
                            continue
        except Exception:
            continue
    return pd.DataFrame(results).drop_duplicates(subset=["url"]) if results else pd.DataFrame()

@st.cache_data(show_spinner="Mencari berita di Kompas...", ttl=300)
def scrape_kompas_fixed(query, max_articles=12):
    data, sess = [], make_session()
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
                    pub, content, title_html = fetch_time_content_title(sess, url)
                    if not pub:
                        ttag = it.select_one(".read__time, .date")
                        if ttag: pub = _parse_id_date_text(ttag.get_text(" ", strip=True))
                    if not pub:
                        m = re.search(r"/(\d{4})/(\d{2})/(\d{2})/(\d{4,8})", url)
                        if m:
                            y, mo, d, hhmm = m.groups()
                            pub = _normalize_to_jakarta(f"{y}-{mo}-{d} {hhmm[:2]}:{hhmm[2:4]}")
                    if not pub: continue
                    if not title or len(title) < 3:
                        title = title_html or slug_to_title(url)
                    if not is_relevant_strict(query, title, "", content, url): continue
                    data.append({
                        "source":"Kompas","title":title,"description":"",
                        "content": (title+" "+content).strip(),
                        "url":url,"publishedAt":pub
                    })
                except Exception:
                    continue
    except Exception:
        pass
    if len(data) < max_articles:
        feeds = [
            "https://nasional.kompas.com/rss","https://internasional.kompas.com/rss",
            "https://ekonomi.kompas.com/rss","https://bola.kompas.com/rss",
            "https://tekno.kompas.com/rss","https://sains.kompas.com/rss",
            "https://megapolitan.kompas.com/rss",
        ]
        try:
            for fu in feeds:
                if len(data) >= max_articles: break
                feed = feedparser.parse(fu)
                for e in feed.entries:
                    if len(data) >= max_articles: break
                    title = getattr(e,"title",""); link = getattr(e,"link",""); summary = getattr(e,"summary","")
                    if not link or not _keywords_ok(title, summary, query):
                        continue
                    pub = ""
                    if getattr(e,"published_parsed",None):
                        utc_dt = datetime(*e.published_parsed[:6], tzinfo=pytz.UTC)
                        pub = utc_dt.astimezone(TZ_JKT).strftime("%Y-%m-%d %H:%M")
                    real, content, title_html = fetch_time_content_title(sess, link)
                    if real: pub = real
                    if not pub: continue
                    if not title or len(title) < 3:
                        title = title_html or slug_to_title(link)
                    if not is_relevant_strict(query, title, summary, content, link): continue
                    data.append({
                        "source":"Kompas","title":title,"description":summary,
                        "content": (title+" "+summary+" "+content).strip(),
                        "url":link,"publishedAt":pub
                    })
        except Exception:
            pass
    return pd.DataFrame(data).drop_duplicates(subset=["url"]) if data else pd.DataFrame()

@st.cache_data(ttl=300)
def scrape_all_sources(query):
    dfs = []
    d1 = scrape_detik(query);  d2 = scrape_cnn_fixed(query); d3 = scrape_kompas_fixed(query)
    if not d1.empty: dfs.append(d1)
    if not d2.empty: dfs.append(d2)
    if not d3.empty: dfs.append(d3)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# ========================= GITHUB HISTORY =========================
@st.cache_resource(ttl=60)
def get_github_client():
    return Github(st.secrets["github_token"])

@st.cache_data(ttl=60)
def load_history_from_github():
    try:
        g = get_github_client()
        repo = g.get_user(st.secrets["repo_owner"]).get_repo(st.secrets["repo_name"])
        contents = repo.get_contents(st.secrets["file_path"])
        data = json.loads(contents.decoded_content.decode("utf-8"))
        if data:
            df = pd.DataFrame(data)
            for col in ["user_id","query","click_time","publishedAt"]:
                if col not in df.columns: df[col] = None
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Gagal memuat riwayat dari GitHub: {e}")
        return pd.DataFrame()

# (Masih ada untuk kompatibilitas batch – tidak dipakai pada mode “klik langsung”)
def save_interaction_to_github(user_id, query, all_articles, clicked_urls):
    g = get_github_client()
    repo = g.get_user(st.secrets["repo_owner"]).get_repo(st.secrets["repo_name"])
    try:
        contents = repo.get_contents(st.secrets["file_path"])
        history_str = contents.decoded_content.decode("utf-8")
        history_list = json.loads(history_str)
    except Exception:
        history_list = []
    now = datetime.now(TZ_JKT).strftime("%A, %d %B %Y %H:%M")
    for _, row in all_articles.iterrows():
        history_list.append({
            "user_id": user_id,
            "query": query,
            "title": str(row.get("title","")),
            "url": str(row.get("url","")),
            "content": str(row.get("content","")),
            "source": str(row.get("source","")),
            "click_time": now,
            "publishedAt": row.get("publishedAt",""),
            "label": 1 if row.get("url","") in clicked_urls else 0
        })
    updated = json.dumps(history_list, indent=2, ensure_ascii=False)
    repo.update_file(st.secrets["file_path"], f"Update history for {query}", updated, contents.sha)

# >>> Simpan 1 klik langsung ke GitHub (aman + retry konflik + tanpa content)
def save_single_click_to_github(user_id: str, query: str, row_like):
    """Append satu interaksi klik ke file history di GitHub."""
    row = dict(row_like)
    now = datetime.now(TZ_JKT).strftime("%A, %d %B %Y %H:%M")
    entry = {
        "user_id": user_id,
        "query": query,
        "title": str(row.get("title","")),
        "url": str(row.get("url","")),
        "source": str(row.get("source","")),
        "click_time": now,
        "publishedAt": row.get("publishedAt",""),
        "label": 1
    }
    g = get_github_client()
    repo = g.get_user(st.secrets["repo_owner"]).get_repo(st.secrets["repo_name"])
    path = st.secrets["file_path"]

    # load + append
    try:
        contents = repo.get_contents(path)
        history = json.loads(contents.decoded_content.decode("utf-8"))
    except Exception:
        contents = None
        history = []

    history.append(entry)
    payload = json.dumps(history, indent=2, ensure_ascii=False)

    # write with conflict retry
    try:
        if contents is None:
            repo.create_file(path, f"Create history for first click: {query}", payload)
        else:
            repo.update_file(path, f"Append click for {query}", payload, contents.sha)
    except GithubException as ge:
        if ge.status == 409:  # refetch sha, try once
            contents = repo.get_contents(path)
            repo.update_file(path, f"Append click (retry) for {query}", payload, contents.sha)
        else:
            raise

# >>> Update Riwayat lokal biar langsung terlihat tanpa reload berat
def append_click_local(user_id: str, query: str, row_like):
    row = dict(row_like)
    now = datetime.now(TZ_JKT).strftime("%A, %d %B %Y %H:%M")
    new_row = {
        "user_id": user_id,
        "query": query,
        "title": str(row.get("title","")),
        "url": str(row.get("url","")),
        "source": str(row.get("source","")),
        "publishedAt": row.get("publishedAt",""),
        "click_time": now,
        "label": 1,
    }
    try:
        if S.history.empty:
            S.history = pd.DataFrame([new_row])
        else:
            S.history = pd.concat([S.history, pd.DataFrame([new_row])], ignore_index=True)
    except Exception:
        pass

# >>> Sanitasi URL sebelum dibuka (hanya http/https, handle //)
def safe_href(u: str) -> str | None:
    if not u:
        return None
    if u.startswith("//"):
        u = "https:" + u
    pu = urllib.parse.urlparse(u)
    if pu.scheme not in ("http", "https"):
        return None
    return u

# ========================= ANALITIK =========================
def get_recent_queries_by_days(user_id, df, days=3):
    if df.empty or "user_id" not in df.columns or "click_time" not in df.columns:
        return {}
    d = df[df["user_id"] == user_id].copy()
    d = d.drop_duplicates(subset=["user_id","query","click_time"])
    d["ts"] = pd.to_datetime(d["click_time"], format="%A, %d %B %Y %H:%M", errors="coerce")
    d["ts"] = d["ts"].fillna(pd.to_datetime(d["click_time"], errors="coerce"))
    d = d.dropna(subset=["ts"])
    now = datetime.now()
    cutoff = now - timedelta(days=days)
    d = d[d["ts"] >= cutoff]
    if d.empty: return {}
    d["date"] = d["ts"].dt.strftime("%d %B %Y")
    grouped = d.groupby("date")["query"].unique().to_dict()
    sorted_dates = sorted(grouped.keys(), key=lambda x: datetime.strptime(x, "%d %B %Y"), reverse=True)
    return {k: grouped[k] for k in sorted_dates}

def trending_by_query_frequency(user_id, df, days=3):
    if df.empty or "user_id" not in df.columns or "query" not in df.columns or "click_time" not in df.columns:
        return []
    d = df[df["user_id"] == user_id].copy()
    d = d.drop_duplicates(subset=["user_id","query","click_time"])
    d["ts"] = pd.to_datetime(d["click_time"], format="%A, %d %B %Y %H:%M", errors="coerce")
    d["ts"] = d["ts"].fillna(pd.to_datetime(d["click_time"], errors="coerce"))
    d = d.dropna(subset=["ts"])
    now = datetime.now()
    cutoff = now - timedelta(days=days)
    d = d[d["ts"] >= cutoff]
    if d.empty:
        return []

    agg = d.groupby("query").agg(
        total=("query", "count"),
        days=("ts", lambda s: s.dt.date.nunique()),
        last_ts=("ts", "max")
    ).reset_index()
    agg = agg.sort_values(by=["days", "total", "last_ts"], ascending=[False, False, False])
    return list(agg[["query", "total"]].itertuples(index=False, name=None))


# ========================= TRAIN & RECOMMEND =========================
def build_training_data(user_id):
    try:
        history_df = load_history_from_github()
        # title wajib, content opsional (biar klik baru tetap kepakai)
        user_data = [
            h for h in history_df.to_dict("records")
            if h.get("user_id") == user_id and "label" in h and h.get("title")
        ]
        df = pd.DataFrame(user_data)
        if df.empty or df["label"].nunique() < 2:
            return pd.DataFrame()
        train, seen = [], set()
        for _, row in df.iterrows():
            text = preprocess_text(str(row.get("title","")) + " " + str(row.get("content","")))
            label = int(row.get("label",0))
            if text and text not in seen:
                train.append({"text":text, "label":label}); seen.add(text)
        return pd.DataFrame(train)
    except Exception:
        return pd.DataFrame()

@st.cache_resource(show_spinner="Melatih model rekomendasi...")
def train_model(df_train):
    X = model_sbert.encode(df_train["text"].tolist())
    y = df_train["label"].tolist()
    if len(set(y)) < 2:
        st.warning("⚠️ Gagal melatih model: hanya ada satu jenis label.")
        return None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    st.sidebar.markdown("📊 **Evaluasi Model:**")
    st.sidebar.write(f"- Akurasi: {accuracy_score(y_test, y_pred):.2f}")
    st.sidebar.write(f"- Presisi: {precision_score(y_test, y_pred):.2f}")
    st.sidebar.write(f"- Recall: {recall_score(y_test, y_pred):.2f}")
    st.sidebar.write(f"- Score F1: {f1_score(y_test, y_pred):.2f}")
    return clf

def recommend(df, query, clf, n_per_source=3, min_score=0.55,
              ensure_all_sources=False, use_lr_boost=True, alpha=0.25,
              per_source_group=True):
    if df.empty: return pd.DataFrame()
    df = df.copy().drop_duplicates(subset=["url"])
    df["processed"] = df.apply(lambda r: preprocess_text(
        f"{r.get('title','')} {r.get('description','')} {r.get('content','')}"
    ), axis=1)
    art_vecs = model_sbert.encode(df["processed"].tolist())
    df["publishedAt_dt"] = pd.to_datetime(df["publishedAt"], errors="coerce")
    df = df.dropna(subset=["publishedAt_dt"])
    if df.empty: return pd.DataFrame()
    q_vec = model_sbert.encode([preprocess_text(query)])
    sims = cosine_similarity(q_vec, art_vecs[:len(df)])[0]
    s_min, s_max = float(sims.min()), float(sims.max())
    sbert_score = (sims - s_min) / (s_max - s_min) if s_max > s_min else sims
    df["sbert_score"] = sbert_score
    if clf is not None and use_lr_boost:
        lr_score = clf.predict_proba(art_vecs[:len(df)])[:, 1]
        bonus = df["title"].apply(lambda t: 0.05 if (t and query.lower() in t.lower()) else 0).values
        final = ((1 - alpha) * df["sbert_score"].values) + (alpha * lr_score) + bonus
    else:
        final = df["sbert_score"].values
    df["final_score"] = final.clip(0, 1)

    filtered = df[df["final_score"] >= min_score].copy()
    if filtered.empty: filtered = df.copy()

    def _top_n(x):
        return x.sort_values(["publishedAt_dt","final_score"], ascending=[False, False]).head(n_per_source)

    if per_source_group:
        got = filtered.groupby("source", group_keys=False).apply(_top_n)
    else:
        got = filtered.sort_values(["publishedAt_dt","final_score"], ascending=[False, False]).head(3*n_per_source)

    return got.sort_values(["publishedAt_dt","final_score"], ascending=[False, False]).reset_index(drop=True)

# ========================= TOMBOL BACA (catat klik + buka artikel) =========================
def _key_for(url, query):
    raw = f"{url}|{query}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]

def render_read_button(url: str, query: str, row_dict: dict, label: str = "Baca selengkapnya"):
    """
    UX halus:
    1) Buka tab baru dulu (user-gesture) → cepat terasa.
    2) Tandai lokal + simpan ke GitHub (tanpa clear cache global).
    """
    btn_key = f"read_{_key_for(url, query)}"
    if st.button(label, key=btn_key):
        # 1) buka tab dulu
        href = safe_href(url)
        if href is None:
            st.warning("Tautan tidak valid atau skemanya tidak diizinkan.")
            return
        safe = json.dumps(href)
        components.html(
            f"""
            <script>
              (function(){{
                try {{
                  var u = {safe};
                  var a = document.createElement('a');
                  a.href = u; a.target = '_blank'; a.rel='noopener noreferrer'; a.style.display='none';
                  document.body.appendChild(a); a.click();
                }} catch(e) {{
                  try {{ window.open({safe}, '_blank'); }} catch(_e) {{}}
                }}
              }})();
            </script>
            """,
            height=0,
        )

        # 2) catat lokal + simpan ke GitHub (tanpa st.cache_data.clear())
        S.clicked_by_query.setdefault(query, set()).add(url)
        append_click_local(USER_ID, query, row_dict)
        try:
            save_single_click_to_github(USER_ID, query, row_dict)
        except Exception as e:
            st.warning(f"Gagal menyimpan history: {e}")

# ========================= APP =========================
def main():
    st.title("📰 SISTEM REKOMENDASI BERITA")
    st.markdown(
        "Sistem ini merekomendasikan berita dari Detik, CNN Indonesia, dan Kompas "
        "berdasarkan riwayat topik serta menyediakan fitur pencarian."
    )

    # Sidebar (hanya cache + model personalisasi)
    if st.sidebar.button("Bersihkan Cache & Muat Ulang"):
        st.cache_data.clear(); st.cache_resource.clear()
        st.success("Cache dibersihkan. Memuat ulang…"); time.sleep(1); st.rerun()

    if S.history.empty:
        S.history = load_history_from_github()

    st.sidebar.header("Model Personalisasi")
    try:
        df_train = build_training_data(USER_ID)
    except Exception as e:
        df_train = pd.DataFrame()
        st.sidebar.warning(f"Gagal menyiapkan data latih: {e}")
    clf = None
    if not df_train.empty and df_train["label"].nunique() > 1:
        st.sidebar.success("Model berhasil dilatih.")
        clf = train_model(df_train)
    else:
        st.sidebar.info("Model belum bisa dilatih karena riwayat tidak mencukupi.")

    # ========== (1) RIWAYAT PENCARIAN BERITA ==========
    st.header("📚 RIWAYAT PENCARIAN BERITA")
    grouped_queries = get_recent_queries_by_days(USER_ID, S.history, days=3)
    if grouped_queries:
        for date, queries in grouped_queries.items():
            st.subheader(f"Tanggal {date}")
            for q in sorted(set(queries)):
                with st.expander(f"- {q}", expanded=True):
                    with st.spinner("Mengambil berita terbaru dari 3 sumber..."):
                        df_latest = scrape_all_sources(q)
                    if df_latest.empty:
                        st.info("❗ Tidak ditemukan berita terbaru untuk topik ini.")
                    else:
                        results_latest = recommend(
                            df_latest, q, clf,
                            n_per_source=3,
                            min_score=DEFAULT_MIN_SCORE,
                            use_lr_boost=USE_LR_BOOST, alpha=ALPHA,
                            per_source_group=PER_SOURCE_GROUP,
                        )
                        if results_latest.empty:
                            st.info("❗ Tidak ada hasil relevan dari portal untuk topik ini.")
                        else:
                            for _, row in results_latest.iterrows():
                                src = get_source_from_url(row["url"])
                                st.markdown(f"**[{src}] {row['title']}**")
                                st.write(row["url"])
                                st.write(f"Waktu: {format_display_time(row.get('publishedAt',''))}")
                                skor = row.get("final_score", row.get("sbert_score", 0.0))
                                render_score_badge(skor)
                                st.markdown("---")
    else:
        st.info("Belum ada riwayat pencarian pada 3 hari terakhir.")

    st.markdown("---")

    # ========== (2) REKOMENDASI BERITA HARI INI ==========
    st.header("🔥 REKOMENDASI BERITA HARI INI")
    trends = trending_by_query_frequency(USER_ID, S.history, days=3)
    if trends:
        q_top, _ = trends[0]
        with st.spinner("Mencari berita..."):
            df_news = scrape_all_sources(q_top)
        if df_news.empty:
            st.info("❗ Tidak ditemukan berita.")
        else:
            results = recommend(
                df_news, q_top, clf,
                n_per_source=1,
                min_score=DEFAULT_MIN_SCORE,
                use_lr_boost=USE_LR_BOOST, alpha=ALPHA,
                per_source_group=PER_SOURCE_GROUP,
            )
            S.trending_results_df = results.copy()
            S.trending_query = q_top
            if results.empty:
                st.info("❗ Tidak ada hasil relevan.")
            else:
                for _, row in results.iterrows():
                    src = get_source_from_url(row["url"])
                    st.markdown(f"**[{src}] {row['title']}**")
                    st.write(f"Waktu: *{format_display_time(row.get('publishedAt',''))}*")
                    skor = row.get("final_score", row.get("sbert_score", 0.0))
                    render_score_badge(skor)
                    render_read_button(row["url"], q_top, row.to_dict())
                    st.markdown("---")
    else:
        st.info("🔥 Tidak ada topik yang sering dicari dalam 3 hari terakhir.")

    st.markdown("---")

    # ========== (3) PENCARIAN BERITA ==========
    st.header("🔍 PENCARIAN BERITA")
    with st.form(key="search_form", clear_on_submit=False):
        search_query = st.text_input("Ketik topik berita yang ingin Anda cari:", value=S.current_query)
        submitted = st.form_submit_button("Cari Berita")

    if submitted:
        if search_query:
            with st.spinner("Mengambil berita dan merekomendasikan..."):
                S.current_search_results = scrape_all_sources(search_query)
                results = recommend(
                    S.current_search_results,
                    search_query, clf,
                    n_per_source=3,
                    min_score=DEFAULT_MIN_SCORE,
                    use_lr_boost=USE_LR_BOOST, alpha=ALPHA,
                    per_source_group=PER_SOURCE_GROUP,
                )
                S.current_recommended_results = results

            S.show_results = True
            S.current_query = search_query
        else:
            st.warning("Mohon masukkan topik pencarian.")

    if S.show_results:
        st.subheader(f"📌 Hasil untuk '{S.current_query}'")
        if S.current_recommended_results.empty:
            st.warning("❗ Tidak ada hasil yang relevan. Coba kata kunci lain.")
        else:
            for _, row in S.current_recommended_results.iterrows():
                src = get_source_from_url(row["url"])
                st.markdown(f"**[{src}] {row['title']}**")
                st.write(f"Waktu: *{format_display_time(row.get('publishedAt',''))}*")
                skor = row.get("final_score", row.get("sbert_score", 0.0))
                render_score_badge(skor)
                render_read_button(row["url"], S.current_query, row.to_dict())
                st.markdown("---")

            clicked_cnt = len(S.clicked_by_query.get(S.current_query, set()))
            total_cnt = len(S.current_recommended_results)
            st.caption(f"Klik tercatat (lokal): {clicked_cnt} dari {total_cnt}. Setiap klik disimpan otomatis ke Riwayat.")

if __name__ == "__main__":
    main()
