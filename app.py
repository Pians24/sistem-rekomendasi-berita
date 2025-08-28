import streamlit as st
import re
import json
import time
import random
import requests
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
from collections import Counter
from github import Github
import streamlit.components.v1 as components

# --- KONFIGURASI HALAMAN STREAMLIT ---
st.set_page_config(page_title="Sistem Rekomendasi Berita", layout="wide")

# --- Inisialisasi Session State ---
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame()
if 'current_search_results' not in st.session_state:
    st.session_state.current_search_results = pd.DataFrame()
if 'show_results' not in st.session_state:
    st.session_state.show_results = False
if 'current_query' not in st.session_state:
    st.session_state.current_query = ""
if 'current_recommended_results' not in st.session_state:
    st.session_state.current_recommended_results = pd.DataFrame()
if 'clicked_urls_in_session' not in st.session_state:
    st.session_state.clicked_urls_in_session = []
if 'url_from_js' not in st.session_state:
    st.session_state.url_from_js = None

# --- Konfigurasi ---
USER_ID = "user_01"
TZ_JKT = pytz.timezone("Asia/Jakarta")

# ====== HTTP session & headers (mirip browser) ======
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
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    })
    return s

def get_source_from_url(url):
    if "detik.com" in url:
        return "Detik"
    elif "cnnindonesia.com" in url:
        return "CNN"
    elif "kompas.com" in url:
        return "Kompas"
    return "Tidak Diketahui"

# === MODEL ===
@st.cache_resource
def load_resources():
    try:
        model_sbert = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    except Exception:
        model_sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return model_sbert
model_sbert = load_resources()

@st.cache_data
def preprocess_text(text):
    text = (text or "").lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# =============== Relevansi ketat (di tahap scrape) ===============
def _keywords(tokens_min3, hay):
    return sum(tok in hay for tok in tokens_min3)

def is_relevant_strict(query: str, title: str, summary: str, content: str, url: str) -> bool:
    """
    - Kalau query punya ≥2 kata (mis. 'tom lembong'), cek frasa persis ada di teks → langsung lolos.
    - Selain itu, butuh ≥2 token (panjang≥3) muncul di gabungan judul/ringkasan/konten/slug URL.
    """
    q = preprocess_text(query)
    hay = preprocess_text(" ".join([title or "", summary or "", content or "", url or ""]))
    # frasa persis (nama orang, entitas 2 kata)
    if len(q.split()) >= 2 and q in hay:
        return True
    # token minimum
    toks = [t for t in re.findall(r"\w+", q) if len(t) >= 3]
    return _keywords(toks, hay) >= 2

# --- Filter ringan untuk RSS (tetap SBERT yang ranking) ---
def _keywords_ok(title: str, summary: str, query: str) -> bool:
    q = preprocess_text(query)
    toks = [t for t in re.findall(r"\w+", q) if len(t) >= 3]
    hay = preprocess_text((title or "") + " " + (summary or ""))
    # RSS cukup minimal 1 token biar nggak terlalu ketat
    return any(tok in hay for tok in toks)

# =================== WAKTU PUBLIKASI ===================
def _has_tz_info(dt_str: str) -> bool:
    if not dt_str:
        return False
    return bool(re.search(r'(Z|[+\-]\d{2}:\d{2}|[+\-]\d{4})$', dt_str.strip()))

@st.cache_data
def _normalize_to_jakarta(dt_str: str) -> str:
    """
    Kembalikan 'YYYY-MM-DD HH:MM' di Asia/Jakarta.
    - Jika string ada timezone (Z/+07:00), parse aware → convert ke Jakarta.
    - Jika string TIDAK ada timezone (naive), anggap itu waktu lokal Jakarta (tanpa geser).
    """
    if not dt_str:
        return ""
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
def _parse_id_date_text(text: str) -> str:
    """Parse teks tanggal Indonesia umum (Detik/CNN/Kompas/20.detik)."""
    if not text:
        return ""
    t = text.strip()

    # "dd/mm/yyyy, HH:MM WIB" (Kompas)
    m0 = re.search(r'(\d{2})/(\d{2})/(\d{4})[, ]+(\d{2}):(\d{2})', t)
    if m0:
        dd, mm, yyyy, hh, mi = m0.groups()
        return _normalize_to_jakarta(f"{yyyy}-{mm}-{dd} {hh}:{mi}")

    bulan_map = {
        "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04", "Mei": "05",
        "Jun": "06", "Jul": "07", "Agu": "08", "Sep": "09", "Okt": "10",
        "Nov": "11", "Des": "12"
    }

    # 'Rabu, 27 Agu 2025 18:47 WIB'
    m1 = re.search(
        r"(Senin|Selasa|Rabu|Kamis|Jumat|Sabtu|Minggu)\s*,\s*(\d{1,2})\s+"
        r"(Jan|Feb|Mar|Apr|Mei|Jun|Jul|Agu|Sep|Okt|Nov|Des)\s+(\d{4})\s+(\d{2}:\d{2})",
        t, flags=re.IGNORECASE
    )
    if m1:
        _, dd, mon, yyyy, hhmm = m1.groups()
        mm = bulan_map.get(mon[:3].title(), "00")
        if mm != "00":
            return _normalize_to_jakarta(f"{yyyy}-{mm}-{int(dd):02d} {hhmm}")

    # 'Rabu, 27/08/2025 18:47 WIB'
    m2 = re.search(
        r"(Senin|Selasa|Rabu|Kamis|Jumat|Sabtu|Minggu)\s*,\s*(\d{2})/(\d{2})/(\d{4})\s+(\d{2}:\d{2})",
        t, flags=re.IGNORECASE
    )
    if m2:
        _, dd, mm, yyyy, hhmm = m2.groups()
        return _normalize_to_jakarta(f"{yyyy}-{mm}-{dd} {hhmm}")

    # ISO-ish di teks
    m3 = re.search(r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2})?(?:Z|[+\-]\d{2}:?\d{2})?)", t)
    if m3:
        return _normalize_to_jakarta(m3.group(1))

    # Relatif
    m4 = re.search(r"(\d+)\s+(menit|jam|hari|minggu)\s+yang\s+lalu", t, flags=re.IGNORECASE)
    if m4:
        jumlah = int(m4.group(1)); unit = m4.group(2).lower()
        now = datetime.now(TZ_JKT)
        if unit.startswith("menit"):
            dt = now - timedelta(minutes=jumlah)
        elif unit.startswith("jam"):
            dt = now - timedelta(hours=jumlah)
        elif unit.startswith("hari"):
            dt = now - timedelta(days=jumlah)
        else:
            dt = now - timedelta(weeks=jumlah)
        return dt.strftime("%Y-%m-%d %H:%M")

    return ""

@st.cache_data
def extract_published_at_from_article_html(art_soup: BeautifulSoup, url: str = "") -> str:
    """
    Urutan ekstraksi:
    1) JSON-LD (datePublished/dateCreated/mainEntityOfPage.datePublished)
    2) <meta property=...> / name=... / itemprop=datePublished
    3) <time datetime="...">
    4) Teks tanggal umum (Detik/CNN/Kompas/20.detik)
    5) Fallback Kompas dari URL /YYYY/MM/DD/HHMM.../
    """
    # 1) JSON-LD
    try:
        for s in art_soup.find_all("script", attrs={"type": "application/ld+json"}):
            raw = (s.string or s.text or "").strip()
            if not raw:
                continue
            data = json.loads(raw)
            candidates = data if isinstance(data, list) else [data]
            for obj in candidates:
                if not isinstance(obj, dict):
                    continue
                for key in ("datePublished", "dateCreated"):
                    if obj.get(key):
                        t = _normalize_to_jakarta(str(obj[key]))
                        if t: return t
                mep = obj.get("mainEntityOfPage")
                if isinstance(mep, dict) and mep.get("datePublished"):
                    t = _normalize_to_jakarta(str(mep["datePublished"]))
                    if t: return t
    except Exception:
        pass

    # 2) META
    meta_candidates = [
        ("property", ["article:published_time", "og:published_time", "og:updated_time"]),
        ("name",     ["publishdate", "pubdate", "DC.date.issued", "date", "content_PublishedDate"]),
        ("itemprop", ["datePublished", "datecreated"])
    ]
    for attr, names in meta_candidates:
        for nm in names:
            tag = art_soup.find("meta", attrs={attr: nm})
            if tag and tag.get("content"):
                t = _normalize_to_jakarta(tag["content"])
                if t: return t

    # 3) <time datetime="...">
    ttag = art_soup.find("time", attrs={"datetime": True})
    if ttag and ttag.get("datetime"):
        t = _normalize_to_jakarta(ttag["datetime"])
        if t: return t

    # 4) Teks tanggal
    for sel in [
        "div.detail__date", "div.read__time", "div.date", "span.date",
        "span.box__date", "div.the_date", "p.date", "time"
    ]:
        tag = art_soup.select_one(sel)
        if tag:
            t = _parse_id_date_text(tag.get_text(" ", strip=True))
            if t: return t

    # 5) Fallback Kompas → dari URL
    if url and "kompas.com" in url:
        m = re.search(r"/(\d{4})/(\d{2})/(\d{2})/(\d{4,8})", url)
        if m:
            y, mo, d, hhmmxx = m.groups()
            hh = hhmmxx[:2]; mi = hhmmxx[2:4]
            if hh.isdigit() and mi.isdigit():
                return _normalize_to_jakarta(f"{y}-{mo}-{d} {hh}:{mi}")

    return ""  # biarkan kosong → caller boleh skip

# ---------- helper untuk ambil waktu dari halaman + konten ringkas ----------
def fetch_time_and_content(sess, link):
    published_at = ""
    content = ""
    try:
        time.sleep(random.uniform(0.4, 0.9))
        ar = sess.get(link, timeout=20)
        if ar.status_code == 200:
            art_soup = BeautifulSoup(ar.content, "html.parser")
            published_at = extract_published_at_from_article_html(art_soup, link)
            paras = art_soup.select("article p, div.read__content p, div.detail__body p, .text-article p")
            content = " ".join([p.get_text(" ", strip=True) for p in paras])[:1500]
    except Exception:
        pass
    return published_at, content

# =================== SCRAPERS ===================
@st.cache_data(show_spinner="Mencari berita di Detik...", ttl=300)
def scrape_detik(query, max_articles=15):
    data = []
    sess = make_session()

    # --- 1) Halaman pencarian Detik ---
    try:
        search_url = f"https://www.detik.com/search/searchall?query={requests.utils.quote(query)}"
        res = sess.get(search_url, timeout=15)
        if res.status_code == 200:
            soup = BeautifulSoup(res.content, "html.parser")
            items = soup.select("article.list-content__item")
            if not items:
                items = soup.select("li.list__item, div.list__item, div.list-content__item")
            for it in items:
                try:
                    a = it.select_one("h3.media__title a, h2.media__title a, a.media__link, a[href]")
                    if not a or not a.get("href"):
                        continue
                    link = a["href"]
                    title = a.get_text(strip=True)
                    desc_el = it.select_one(".media__desc, .desc, p")
                    description = desc_el.get_text(strip=True) if desc_el else ""

                    published_at, content = fetch_time_and_content(sess, link)
                    if not published_at:
                        dt_hint = it.select_one(".media__date, .date")
                        if dt_hint:
                            published_at = _parse_id_date_text(dt_hint.get_text(" ", strip=True))
                    if not published_at:
                        continue

                    # >>> relevansi ketat
                    if not is_relevant_strict(query, title, description, content, link):
                        continue

                    data.append({
                        "source": "Detik",
                        "title": title,
                        "description": description,
                        "content": (title + " " + description + " " + content).strip(),
                        "url": link,
                        "publishedAt": published_at
                    })
                    if len(data) >= max_articles:
                        break
                except Exception:
                    continue
    except Exception:
        pass

    # --- 2) Fallback RSS Detik (dengan filter ringan + konfirmasi waktu dari halaman) ---
    if len(data) < max_articles:
        feeds = [
            "https://news.detik.com/rss",
            "https://finance.detik.com/rss",
            "https://sport.detik.com/rss",
            "https://hot.detik.com/rss",
            "https://inet.detik.com/rss",
            "https://health.detik.com/rss",
            "https://oto.detik.com/rss",
            "https://travel.detik.com/rss",
            "https://food.detik.com/rss",
            "https://20.detik.com/rss",
        ]
        try:
            for fu in feeds:
                if len(data) >= max_articles:
                    break
                feed = feedparser.parse(fu)
                for e in feed.entries:
                    if len(data) >= max_articles:
                        break
                    title = getattr(e, "title", "")
                    link = getattr(e, "link", "")
                    summary = getattr(e, "summary", "")
                    if not link or not _keywords_ok(title, summary, query):
                        continue

                    published_at = ""
                    if getattr(e, "published_parsed", None):
                        utc_dt = datetime(*e.published_parsed[:6], tzinfo=pytz.UTC)
                        published_at = utc_dt.astimezone(TZ_JKT).strftime("%Y-%m-%d %H:%M")
                    real, content = fetch_time_and_content(sess, link)
                    if real: published_at = real
                    if not published_at:
                        continue

                    # relevansi ketat di sini juga
                    if not is_relevant_strict(query, title, summary, content, link):
                        continue

                    data.append({
                        "source": "Detik",
                        "title": title,
                        "description": summary,
                        "content": (title + " " + summary + " " + content).strip(),
                        "url": link,
                        "publishedAt": published_at
                    })
        except Exception:
            pass

    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data).drop_duplicates(subset=["url"])

@st.cache_data(show_spinner="Mencari berita di CNN...", ttl=300)
def scrape_cnn_fixed(query, max_results=12):
    urls_to_scrape = [
        f"https://www.cnnindonesia.com/search?query={requests.utils.quote(query)}",
        "https://www.cnnindonesia.com/nasional/rss",
        "https://www.cnnindonesia.com/internasional/rss",
        "https://www.cnnindonesia.com/ekonomi/rss",
        "https://www.cnnindonesia.com/olahraga/rss",
        "https://www.cnnindonesia.com/gaya-hidup/rss",
    ]
    results = []
    sess = make_session()

    for url in urls_to_scrape:
        if len(results) >= max_results:
            break
        try:
            if "rss" in url:
                feed = feedparser.parse(url)
                for entry in feed.entries:
                    if len(results) >= max_results:
                        break
                    title = getattr(entry, "title", "")
                    link = getattr(entry, "link", "")
                    summary = getattr(entry, "summary", "")
                    if not link or not _keywords_ok(title, summary, query):
                        continue
                    pub = ""
                    if getattr(entry, "published_parsed", None):
                        utc_dt = datetime(*entry.published_parsed[:6], tzinfo=pytz.UTC)
                        pub = utc_dt.astimezone(TZ_JKT).strftime("%Y-%m-%d %H:%M")

                    real, content = fetch_time_and_content(sess, link)
                    if real: pub = real
                    if not pub:
                        continue

                    # relevansi ketat
                    if not is_relevant_strict(query, title, summary, content, link):
                        continue

                    results.append({
                        "source": "CNN",
                        "title": title,
                        "description": summary,
                        "content": (title + " " + summary + " " + content).strip(),
                        "url": link,
                        "publishedAt": pub
                    })
            else:
                res = sess.get(url, timeout=15)
                if res.status_code == 200:
                    soup = BeautifulSoup(res.content, 'html.parser')
                    articles_raw = soup.find_all('article', class_='box--card')
                    for article in articles_raw:
                        if len(results) >= max_results:
                            break
                        try:
                            link_tag = article.find('a', class_='box--card__link')
                            if not link_tag:
                                continue
                            link = link_tag['href']
                            ttl_el = link_tag.find('span', class_='box--card__title')
                            title = ttl_el.get_text(strip=True) if ttl_el else ""
                            sum_el = article.find('span', class_='box--card__desc')
                            summary = sum_el.get_text(strip=True) if sum_el else ""

                            pub, content = fetch_time_and_content(sess, link)
                            if not pub:
                                continue

                            # relevansi ketat
                            if not is_relevant_strict(query, title, summary, content, link):
                                continue

                            results.append({
                                "source": get_source_from_url(link),
                                "title": title,
                                "description": summary,
                                "content": (title + " " + summary + " " + content).strip(),
                                "url": link,
                                "publishedAt": pub
                            })
                        except Exception:
                            continue
        except Exception:
            continue
    return pd.DataFrame(results).drop_duplicates(subset=["url"]) if results else pd.DataFrame()

@st.cache_data(show_spinner="Mencari berita di Kompas...", ttl=300)
def scrape_kompas_fixed(query, max_articles=12):
    data = []
    sess = make_session()

    # --- 1) Halaman pencarian Kompas ---
    try:
        search_url = f"https://search.kompas.com/search?q={requests.utils.quote(query)}"
        res = sess.get(search_url, timeout=20)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            items = soup.select("div.articleItem")
            if not items:
                items = soup.select("div.article__list, li.article__item, div.iso__item")
            for it in items[:max_articles]:
                try:
                    a = it.select_one("a.article-link, a.article__link, a[href]")
                    h = it.select_one("h2.articleTitle, h3.article__title, h2.article__title")
                    if not a or not h:
                        continue
                    url = a.get("href", "")
                    title = h.get_text(strip=True)
                    if not url or "kompas.com" not in url:
                        continue

                    pub, content = fetch_time_and_content(sess, url)
                    if not pub:
                        ttag = it.select_one(".read__time, .date")
                        if ttag:
                            pub = _parse_id_date_text(ttag.get_text(" ", strip=True))
                    if not pub:
                        m = re.search(r"/(\d{4})/(\d{2})/(\d{2})/(\d{4,8})", url)
                        if m:
                            y, mo, d, hhmm = m.groups()
                            pub = _normalize_to_jakarta(f"{y}-{mo}-{d} {hhmm[:2]}:{hhmm[2:4]}")
                    if not pub:
                        continue

                    # relevansi ketat
                    if not is_relevant_strict(query, title, "", content, url):
                        continue

                    data.append({
                        "source": "Kompas",
                        "title": title,
                        "description": "",
                        "content": (title + " " + content).strip(),
                        "url": url,
                        "publishedAt": pub
                    })
                except Exception:
                    continue
    except Exception:
        pass

    # --- 2) Fallback RSS Kompas ---
    if len(data) < max_articles:
        feeds = [
            "https://nasional.kompas.com/rss",
            "https://internasional.kompas.com/rss",
            "https://ekonomi.kompas.com/rss",
            "https://bola.kompas.com/rss",
            "https://tekno.kompas.com/rss",
            "https://sains.kompas.com/rss",
            "https://megapolitan.kompas.com/rss",
        ]
        try:
            for fu in feeds:
                if len(data) >= max_articles:
                    break
                feed = feedparser.parse(fu)
                for e in feed.entries:
                    if len(data) >= max_articles:
                        break
                    title = getattr(e, "title", "")
                    link = getattr(e, "link", "")
                    summary = getattr(e, "summary", "")
                    if not link or not _keywords_ok(title, summary, query):
                        continue
                    pub = ""
                    if getattr(e, "published_parsed", None):
                        utc_dt = datetime(*e.published_parsed[:6], tzinfo=pytz.UTC)
                        pub = utc_dt.astimezone(TZ_JKT).strftime("%Y-%m-%d %H:%M")
                    real, content = fetch_time_and_content(sess, link)
                    if real: pub = real
                    if not pub:
                        continue

                    # relevansi ketat
                    if not is_relevant_strict(query, title, summary, content, link):
                        continue

                    data.append({
                        "source": "Kompas",
                        "title": title,
                        "description": summary,
                        "content": (title + " " + summary + " " + content).strip(),
                        "url": link,
                        "publishedAt": pub
                    })
        except Exception:
            pass

    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data).drop_duplicates(subset=["url"])

@st.cache_data(show_spinner="Menggabungkan hasil...", ttl=300)
def scrape_all_sources(query):
    dfs = []
    df_detik = scrape_detik(query)
    if not df_detik.empty:
        dfs.append(df_detik)
    df_cnn = scrape_cnn_fixed(query)
    if not df_cnn.empty:
        dfs.append(df_cnn)
    df_kompas = scrape_kompas_fixed(query)
    if not df_kompas.empty:
        dfs.append(df_kompas)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

# =================== RIWAYAT (GitHub) ===================
@st.cache_resource(ttl=60)
def get_github_client():
    return Github(st.secrets["github_token"])

@st.cache_data(ttl=60)
def load_history_from_github():
    try:
        g = get_github_client()
        repo = g.get_user(st.secrets["repo_owner"]).get_repo(st.secrets["repo_name"])
        contents = repo.get_contents(st.secrets["file_path"])
        file_content = contents.decoded_content.decode('utf-8')
        data = json.loads(file_content)
        if data:
            df = pd.DataFrame(data)
            required_cols = ['user_id', 'query', 'click_time', 'publishedAt']
            for col in required_cols:
                if col not in df.columns:
                    df[col] = None
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Gagal memuat riwayat dari GitHub: {e}")
        return pd.DataFrame()

def save_interaction_to_github(user_id, query, all_articles, clicked_urls):
    g = get_github_client()
    repo = g.get_user(st.secrets["repo_owner"]).get_repo(st.secrets["repo_name"])
    try:
        contents = repo.get_contents(st.secrets["file_path"])
        history_str = contents.decoded_content.decode('utf-8')
        history_list = json.loads(history_str)
    except Exception:
        history_list = []
    now = datetime.now(TZ_JKT).strftime("%A, %d %B %Y %H:%M")
    for _, row in all_articles.iterrows():
        history_list.append({
            "user_id": user_id,
            "query": query,
            "title": str(row.get('title', '')),
            "url": str(row.get('url', '')),
            "content": str(row.get('content', '')),
            "source": str(row.get('source', '')),
            "click_time": now,
            "publishedAt": row.get('publishedAt', ""),
            "label": 1 if row.get('url', '') in clicked_urls else 0
        })
    updated_content = json.dumps(history_list, indent=2, ensure_ascii=False)
    repo.update_file(
        st.secrets["file_path"],
        f"Update history for {query}",
        updated_content,
        contents.sha
    )

# =================== ANALITIK RIWAYAT ===================
def get_recent_queries_by_days(user_id, df, days=3):
    if df.empty or "user_id" not in df.columns or 'click_time' not in df.columns:
        return {}
    df_user = df[df["user_id"] == user_id].copy()
    if 'publishedAt' not in df_user.columns:
        df_user['publishedAt'] = df_user['click_time']
    df_user['date_to_process'] = df_user['click_time']
    df_user["timestamp"] = pd.to_datetime(
        df_user["date_to_process"], format="%A, %d %B %Y %H:%M", errors='coerce'
    )
    df_user["timestamp"].fillna(pd.to_datetime(
        df_user["date_to_process"], format="%Y-%m-%d %H:%M", errors='coerce'
    ), inplace=True)
    df_user = df_user.dropna(subset=['timestamp']).copy()
    try:
        df_user['timestamp'] = pd.to_datetime(df_user['timestamp']).dt.tz_localize(TZ_JKT, ambiguous='NaT', nonexistent='NaT')
    except Exception:
        return {}
    df_user = df_user.dropna(subset=['timestamp']).copy()
    if df_user.empty:
        return {}
    now = datetime.now(TZ_JKT)
    cutoff = now - timedelta(days=days)
    recent_df = df_user[df_user["timestamp"] >= cutoff].copy()
    if recent_df.empty:
        return {}
    recent_df.loc[:, 'date'] = recent_df['timestamp'].dt.strftime('%d %B %Y')
    grouped = recent_df.groupby('date')['query'].unique().to_dict()
    sorted_dates = sorted(grouped.keys(), key=lambda d: datetime.strptime(d, '%d %B %Y'), reverse=True)
    return {d: grouped[d] for d in sorted_dates}

def get_most_frequent_topics(user_id, df, days=3):
    if df.empty or "user_id" not in df.columns:
        return []
    df_user = df[df["user_id"] == user_id].copy()
    if 'publishedAt' not in df_user.columns:
        df_user['publishedAt'] = df_user['click_time']
    df_user['date_to_process'] = df_user['publishedAt']
    df_user["timestamp"] = pd.to_datetime(df_user["date_to_process"], format="%Y-%m-%d %H:%M", errors='coerce')
    df_user["timestamp"].fillna(pd.to_datetime(
        df_user["date_to_process"], format="%A, %d %B %Y %H:%M", errors='coerce'
    ), inplace=True)
    df_user = df_user.dropna(subset=['timestamp'])
    now = datetime.now(TZ_JKT)
    cutoff = now - timedelta(days=days)
    recent_df = df_user[df_user["timestamp"] >= cutoff]
    if recent_df.empty:
        return []
    counts = Counter(recent_df['query'])
    return sorted(counts.items(), key=lambda x: x[1], reverse=True)

def build_training_data(user_id):
    history_df = load_history_from_github()
    user_data = [h for h in history_df.to_dict('records')
                 if h.get("user_id") == user_id and "label" in h and h.get("title") and h.get("content")]
    df = pd.DataFrame(user_data)
    if df.empty or df["label"].nunique() < 2:
        return pd.DataFrame()
    train = []
    seen = set()
    for _, row in df.iterrows():
        text = preprocess_text(str(row.get("title", "")) + " " + str(row.get("content", "")))
        label = int(row.get("label", 0))
        if text and text not in seen:
            train.append({"text": text, "label": label})
            seen.add(text)
    return pd.DataFrame(train)

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
    st.sidebar.write(f"- Skor F1: {f1_score(y_test, y_pred):.2f}")
    return clf

# =================== REKOMENDASI (SBERT) ===================
def recommend(df, query, clf, n_per_source=3, min_score=0.55, ensure_all_sources=False):
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.drop_duplicates(subset=['url'], inplace=True)

    df["processed"] = df.apply(lambda r: preprocess_text(
        (r.get('title', '') or '') + ' ' +
        (r.get('description', '') or '') + ' ' +
        (r.get('content', '') or '')
    ), axis=1)
    vec = model_sbert.encode(df["processed"].tolist())

    df['publishedAt_dt'] = pd.to_datetime(df['publishedAt'], errors='coerce')
    df['vec_temp'] = list(vec)
    df = df.dropna(subset=['publishedAt_dt'])
    vec = df['vec_temp'].tolist()
    df = df.drop(columns=['vec_temp'])
    if df.empty:
        return pd.DataFrame()

    # Ranking
    if clf:
        scores = clf.predict_proba(vec)[:, 1]
        df["score"] = scores
        df["bonus"] = df["title"].apply(lambda x: 0.1 if (x and query.lower() in x.lower()) else 0)
        df["final_score"] = (df["score"] + df["bonus"]).clip(0, 1)
        filtered = df[df['final_score'] >= min_score].copy()
        if filtered.empty:
            filtered = df.copy()

        def top_n(x):
            return x.sort_values(by=['publishedAt_dt', 'final_score'], ascending=[False, False]).head(n_per_source)
        got = filtered.groupby("source", group_keys=False).apply(top_n, include_groups=False)
    else:
        q_vec = model_sbert.encode([preprocess_text(query)])
        sims = cosine_similarity(q_vec, vec)[0]
        df["similarity"] = sims

        filtered = df[df['similarity'] >= min_score].copy()
        if filtered.empty:
            filtered = df.copy()

        def top_n_sim(x):
            return x.sort_values(by=['publishedAt_dt', 'similarity'], ascending=[False, False]).head(n_per_source)
        got = filtered.groupby("source", group_keys=False).apply(top_n_sim, include_groups=False)

    got = got.sort_values(by=['publishedAt_dt'], ascending=False).reset_index(drop=True)
    # NOTE: ensure_all_sources=False by default agar tidak memaksa artikel tidak relevan masuk
    return got

def format_display_time(display_time: str) -> str:
    try:
        dt_obj = datetime.strptime(display_time, "%Y-%m-%d %H:%M")
        return dt_obj.strftime("%Y-%m-%d %H:%M") if dt_obj.strftime("%H:%M") != "00:00" else dt_obj.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return display_time or "Tidak Diketahui"

# =================== APP ===================
def handle_js_click(url):
    if url not in st.session_state.clicked_urls_in_session:
        st.session_state.clicked_urls_in_session.append(url)
        st.rerun()

def main():
    st.title("📰 Rekomendasi Berita")
    st.markdown("Aplikasi ini merekomendasikan berita dari Detik, CNN, dan Kompas berdasarkan riwayat dan topik Anda. Waktu publikasi diambil langsung dari halaman artikel.")

    if st.sidebar.button("Bersihkan Cache & Muat Ulang"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cache dibersihkan. Memuat ulang…")
        time.sleep(1); st.rerun()

    if st.session_state.history.empty:
        st.session_state.history = load_history_from_github()

    st.sidebar.header("Model Personalisasi")
    df_train = build_training_data(USER_ID)
    clf = None
    if not df_train.empty and df_train['label'].nunique() > 1:
        st.sidebar.success("Model berhasil dilatih.")
        clf = train_model(df_train)
    else:
        st.sidebar.info("Model belum bisa dilatih karena riwayat tidak mencukupi.")

    # ==== PENCARIAN BERITA PER TANGGAL ====
    st.header("📚 Pencarian Berita per Tanggal")
    grouped_queries = get_recent_queries_by_days(USER_ID, st.session_state.history, days=3)

    if grouped_queries:
        for date, queries in grouped_queries.items():
            st.subheader(f"Tanggal {date}")
            unique_queries = sorted(list(set(queries)))
            for q in unique_queries:
                with st.expander(f"- {q}", expanded=False):
                    with st.spinner('Mengambil berita terbaru dari 3 sumber...'):
                        df_latest = scrape_all_sources(q)

                    if df_latest.empty:
                        st.info("❗ Tidak ditemukan berita terbaru untuk topik ini.")
                        continue

                    # tampilkan ringkas jumlah per sumber (setelah filter relevansi di scraper)
                    cnt = df_latest['source'].value_counts().to_dict()
                    st.caption(f"Hasil (relevan): Detik={cnt.get('Detik',0)} | CNN={cnt.get('CNN',0)} | Kompas={cnt.get('Kompas',0)}")

                    # rekomendasi SBERT tanpa memaksa 3 per sumber (biar relevan aja yang keluar)
                    results_latest = recommend(df_latest, q, clf, n_per_source=3, min_score=0.55, ensure_all_sources=False)

                    if results_latest.empty:
                        st.info("❗ Tidak ada hasil relevan dari portal untuk topik ini.")
                        continue

                    for _, row in results_latest.iterrows():
                        source_name = get_source_from_url(row['url'])
                        st.markdown(f"**[{source_name}]** {row['title']}")
                        st.markdown(f"[{row['url']}]({row['url']})")
                        st.write(f"Waktu Publikasi: *{format_display_time(row.get('publishedAt', 'Tidak Diketahui'))}*")
                        skor_key = 'final_score' if 'final_score' in row else ('similarity' if 'similarity' in row else None)
                        if skor_key:
                            st.write(f"Skor: `{row[skor_key]:.2f}`")
                        st.markdown("---")
    else:
        st.info("Belum ada riwayat pencarian pada 3 hari terakhir.")

    st.markdown("---")

    # ==== REKOMENDASI BERITA HARI INI ====
    st.header("🔥 Rekomendasi Berita Hari Ini")
    most_frequent_topics = get_most_frequent_topics(USER_ID, st.session_state.history, days=3)
    if most_frequent_topics:
        q, _ = most_frequent_topics[0]
        with st.spinner('Mencari berita...'):
            df_news = scrape_all_sources(q)
        if df_news.empty:
            st.info("❗ Tidak ditemukan berita.")
        else:
            results = recommend(df_news, q, clf, n_per_source=1, min_score=0.55, ensure_all_sources=False)
            if results.empty:
                st.info("❗ Tidak ada hasil relevan.")
            else:
                for _, row in results.iterrows():
                    source_name = get_source_from_url(row['url'])
                    st.markdown(f"**[{source_name}]** {row['title']}")
                    st.markdown(row['url'])
                    st.write(f"Waktu: *{format_display_time(row.get('publishedAt', 'Tidak Diketahui'))}*")
                    skor_key = 'final_score' if 'final_score' in row else 'similarity'
                    st.write(f"Skor: `{row[skor_key]:.2f}`")
                    st.markdown("---")
    else:
        st.info("🔥 Tidak ada topik yang sering dicari dalam 3 hari terakhir.")

    st.markdown("---")

    # ==== PENCARIAN BEBAS ====
    st.header("🔍 Pencarian Berita")
    search_query = st.text_input("Ketik topik berita yang ingin Anda cari:", key="search_input")
    if st.button("Cari Berita"):
        if search_query:
            if 'current_query' in st.session_state and st.session_state.current_query:
                save_interaction_to_github(
                    USER_ID, st.session_state.current_query,
                    st.session_state.current_recommended_results,
                    st.session_state.clicked_urls_in_session
                )
                st.cache_data.clear()
                st.session_state.history = load_history_from_github()
            with st.spinner('Mengambil berita dan merekomendasikan...'):
                st.session_state.current_search_results = scrape_all_sources(search_query)
                results = recommend(
                    st.session_state.current_search_results,
                    search_query, clf, n_per_source=3, min_score=0.55, ensure_all_sources=False
                )
                st.session_state.current_recommended_results = results
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
                source_name = get_source_from_url(row['url'])
                button_html = f"""
<style>
.styled-button {{
  background-color: #007bff; color: white; padding: 10px 20px;
  text-align: center; text-decoration: none; display: inline-block;
  font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 8px; border: none;
}}
</style>
<button class="styled-button"
  onclick="window.parent.postMessage({{ streamlit: true, event: 'st_event', data: {{ url: '{row['url']}' }} }}, '*');
           window.open('{row['url']}', '_blank');">
  Buka Artikel & Catat Interaksi
</button>
"""
                st.markdown(f"**[{source_name}]** {row['title']}")
                st.markdown(f"[{row['url']}]({row['url']})")
                st.write(f"Waktu: *{format_display_time(row.get('publishedAt', 'Tidak Diketahui'))}*")
                skor_key = 'final_score' if 'final_score' in row else 'similarity'
                st.write(f"Skor: `{row[skor_key]:.2f}`")
                st.markdown(button_html, unsafe_allow_html=True)
                st.markdown("---")
        if st.session_state.current_query:
            st.info(f"Anda telah mencatat {len(st.session_state.clicked_urls_in_session)} artikel. Data akan disimpan saat Anda memulai pencarian baru.")

def on_message(message):
    if 'url' in message:
        st.session_state.url_from_js = message['url']

components.html("""
<script>
  window.addEventListener('message', event => {
    if (event.data && event.data.streamlit && event.data.event === 'st_event') {
      window.parent.postMessage(event.data, '*');
    }
  });
</script>
""", height=0, width=0)

if st.session_state.url_from_js:
    handle_js_click(st.session_state.url_from_js)
    st.session_state.url_from_js = None

if __name__ == "__main__":
    main()
