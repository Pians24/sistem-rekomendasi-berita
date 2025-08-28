import streamlit as st
import os
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
import base64
import streamlit.components.v1 as components

# --- KONFIGURASI HALAMAN STREAMLIT ---
st.set_page_config(page_title="Sistem Rekomendasi Berita", layout="wide")

# --- Inisialisasi Session State di awal skrip ---
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

# --- Konfigurasi dan Inisialisasi Lainnya ---
USER_ID = "user_01"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0"}

def get_source_from_url(url):
    if "detik.com" in url:
        return "Detik"
    elif "cnnindonesia.com" in url:
        return "CNN"
    elif "kompas.com" in url:
        return "Kompas"
    return "Tidak Diketahui"

# === SUMBER DAYA: SBERT saja (tanpa NLTK) ===
@st.cache_resource
def load_resources():
    try:
        model_sbert = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    except Exception:
        model_sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return model_sbert

model_sbert = load_resources()

# 2. Fungsi Pra-pemrosesan (tanpa stopwords)
@st.cache_data
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 3. Ekstrak waktu (DIPERBAIKI)
@st.cache_data
def extract_datetime_from_title(title, url=None):
    zona = pytz.timezone("Asia/Jakarta")
    now = datetime.now(zona)

    # Pola untuk Detik: "Rabu, 27 Agu 2025 18:47 WIB"
    bulan_map_detik = {
        "Agu": "08", "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04", "Mei": "05",
        "Jun": "06", "Jul": "07", "Sep": "09", "Okt": "10", "Nov": "11", "Des": "12"
    }
    match_detik_date = re.search(r"(\w+, \d{1,2}) (\w+) (\d{4}) (\d{2}:\d{2})", title)
    if match_detik_date:
        day_str, day, month_str, year, time_str = match_detik_date.groups()
        month_num = bulan_map_detik.get(month_str, "00")
        if month_num != "00":
            try:
                dt_obj = datetime.strptime(f"{year}-{month_num}-{day} {time_str}", "%Y-%m-%d %H:%M")
                return zona.localize(dt_obj).strftime("%Y-%m-%d %H:%M")
            except ValueError:
                pass

    # Pola untuk CNN: "Rabu, 27/08/2025 18:47 WIB"
    match_cnn_date = re.search(r"(\w+), (\d{2})/(\d{2})/(\d{4}) (\d{2}:\d{2})", title)
    if match_cnn_date:
        day_str, day, month, year, time_str = match_cnn_date.groups()
        try:
            dt_obj = datetime.strptime(f"{year}-{month}-{day} {time_str}", "%Y-%m-%d %H:%M")
            return zona.localize(dt_obj).strftime("%Y-%m-%d %H:%M")
        except ValueError:
            pass

    # Pola untuk Kompas di URL: /2025/08/25/11480341/
    if url and "kompas.com" in url:
        url_match = re.search(r"/(\d{4})/(\d{2})/(\d{2})/(\d{2})(\d{2})", url)
        if url_match:
            y, m, d, h, mi = url_match.groups()
            try:
                dt = datetime.strptime(f"{y}-{m}-{d} {h}:{mi}", "%Y-%m-%d %H:%M")
                return zona.localize(dt).strftime("%Y-%m-%d %H:%M")
            except:
                pass

    # Pola untuk format lain (sebagai fallback)
    match_absolute = re.search(r"(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2})", title)
    if match_absolute:
        try:
            dt = datetime.strptime(match_absolute.group(1), "%Y-%m-%d %H:%M")
            return zona.localize(dt).strftime("%Y-%m-%d %H:%M")
        except:
            pass
            
    # Pola untuk "X waktu yang lalu" (sebagai fallback terakhir)
    match_relative = re.search(r"(\d+)\s+(menit|jam|hari|minggu)\s+yang lalu", title, re.IGNORECASE)
    if match_relative:
        jumlah, satuan = match_relative.groups()
        jumlah = int(jumlah)
        if "menit" in satuan:
            dt = now - timedelta(minutes=jumlah)
        elif "jam" in satuan:
            dt = now - timedelta(hours=jumlah)
        elif "hari" in satuan:
            dt = now - timedelta(days=jumlah)
        elif "minggu" in satuan:
            dt = now - timedelta(weeks=jumlah)
        return dt.strftime("%Y-%m-%d %H:%M")

    return ""

@st.cache_data
def is_relevant(title, query, content="", threshold=0.5):
    combined = f"{title} {content}"
    combined_vecs = model_sbert.encode([combined])
    query_vecs = model_sbert.encode([query])
    sim = cosine_similarity(combined_vecs, query_vecs)[0][0]
    return sim >= threshold

# 4. Scraper per sumber (DIPERBAIKI)
@st.cache_data(show_spinner="Mencari berita di Detik...")
def scrape_detik(query, max_articles=15):
    url = f"https://www.detik.com/search/searchall?query={query.replace(' ', '+')}"
    data = []
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
    ]
    headers_main = {"User-Agent": random.choice(USER_AGENTS)}

    try:
        res = requests.get(url, headers=headers_main, timeout=10)
        if res.status_code == 200:
            soup = BeautifulSoup(res.content, "html.parser")
            articles_raw = soup.select("article.list-content__item")
            if not articles_raw:
                st.warning("Scraper Detik tidak menemukan artikel di halaman pencarian.")
                return pd.DataFrame()
                
            for article in articles_raw:
                try:
                    title_tag = article.find('h3', class_='media__title')
                    link = title_tag.a['href'] if title_tag and title_tag.a else ''
                    description_tag = article.find('div', class_='media__desc')

                    if not title_tag or not link:
                        continue

                    title = title_tag.get_text(strip=True)
                    description = description_tag.get_text(strip=True) if description_tag else ""
                    published_at = ''

                    if is_relevant(title, query, description):
                        time.sleep(random.uniform(1, 2))
                        headers_article = {"User-Agent": random.choice(USER_AGENTS)}
                        art_res = requests.get(link, headers=headers_article, timeout=10)
                        if art_res.status_code == 200:
                            art_soup = BeautifulSoup(art_res.content, 'html.parser')
                            date_tag = art_soup.find('div', class_='detail__date')
                            if date_tag:
                                date_text = date_tag.get_text(strip=True)
                                published_at = extract_datetime_from_title(date_text, link)

                        if not published_at:
                            jakarta_tz = pytz.timezone("Asia/Jakarta")
                            published_at = datetime.now(jakarta_tz).strftime("%Y-%m-%d %H:%M")

                        data.append({
                            "source": get_source_from_url(link),
                            "title": title,
                            "description": description,
                            "content": f"{title} {description}",
                            "url": link,
                            "publishedAt": published_at
                        })
                except Exception as e:
                    continue
                if len(data) >= max_articles:
                    break
            return pd.DataFrame(data)
        else:
            st.warning(f"Gagal scraping Detik: Status code {res.status_code}. Mengembalikan DataFrame kosong.")
    except (requests.exceptions.RequestException, Exception) as e:
        st.warning(f"Gagal scraping Detik: {e}. Mengembalikan DataFrame kosong.")
        time.sleep(random.uniform(1, 3))
    return pd.DataFrame()

@st.cache_data(show_spinner="Mencari berita di CNN...")
def scrape_cnn_fixed(query, max_results=10):
    url = f"https://www.cnnindonesia.com/search?query={query.replace(' ', '+')}"
    results = []
    
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
    ]
    headers_main = {"User-Agent": random.choice(USER_AGENTS)}

    try:
        res = requests.get(url, headers=headers_main, timeout=10)
        if res.status_code == 200:
            soup = BeautifulSoup(res.content, 'html.parser')
            articles_raw = soup.find_all('article', class_='box--card')
            
            for article in articles_raw:
                try:
                    link_tag = article.find('a', class_='box--card__link')
                    if not link_tag:
                        continue
                    
                    link = link_tag['href']
                    title = link_tag.find('span', class_='box--card__title').get_text(strip=True) if link_tag.find('span', class_='box--card__title') else ""
                    summary = article.find('span', class_='box--card__desc').get_text(strip=True) if article.find('span', class_='box--card__desc') else ""
                    
                    if is_relevant(title, query, summary):
                        time.sleep(random.uniform(1, 2))
                        headers_article = {"User-Agent": random.choice(USER_AGENTS)}
                        art_res = requests.get(link, headers=headers_article, timeout=10)
                        published_at = ''
                        if art_res.status_code == 200:
                            art_soup = BeautifulSoup(art_res.content, 'html.parser')
                            date_tag = art_soup.find('div', class_='detail__date')
                            if date_tag:
                                date_text = date_tag.get_text(strip=True)
                                published_at = extract_datetime_from_title(date_text, link)
                            
                        if not published_at:
                            jakarta_tz = pytz.timezone("Asia/Jakarta")
                            published_at = datetime.now(jakarta_tz).strftime("%Y-%m-%d %H:%M")
                        
                        results.append({
                            "source": get_source_from_url(link),
                            "title": title,
                            "description": summary,
                            "content": f"{title} {summary}",
                            "url": link,
                            "publishedAt": published_at
                        })
                    if len(results) >= max_results:
                        return pd.DataFrame(results)
                except Exception as e:
                    continue
            return pd.DataFrame(results)
        else:
            st.warning(f"Gagal scraping CNN: Status code {res.status_code}. Mengembalikan DataFrame kosong.")
    except (requests.exceptions.RequestException, Exception) as e:
        st.warning(f"Gagal scraping CNN: {e}. Mengembalikan DataFrame kosong.")
        time.sleep(random.uniform(1, 3))
    return pd.DataFrame()

@st.cache_data(show_spinner="Mencari berita di Kompas...")
def scrape_kompas_fixed(query, max_articles=10):
    search_url = f"https://search.kompas.com/search?q={query.replace(' ', '+')}"
    data = []
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
    ]
    headers_main = {"User-Agent": random.choice(USER_AGENTS)}

    try:
        res = requests.get(search_url, headers=headers_main, timeout=10)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            articles_raw = soup.select("div.articleItem")[:max_articles]
            
            if not articles_raw:
                st.warning("Scraper Kompas tidak menemukan artikel di halaman pencarian.")
                return pd.DataFrame()

            for item in articles_raw:
                try:
                    a_tag = item.select_one("a.article-link")
                    title_tag = item.select_one("h2.articleTitle")

                    if not a_tag or not title_tag:
                        continue

                    url = a_tag["href"]
                    title = title_tag.get_text(strip=True)

                    time.sleep(random.uniform(2, 5))
                    headers_article = {"User-Agent": random.choice(USER_AGENTS)}
                    art_res = requests.get(url, headers=headers_article, timeout=15)
                    
                    if art_res.status_code != 200:
                        continue

                    art_soup = BeautifulSoup(art_res.text, "html.parser")
                    content_paras = art_soup.select("div.read__content > p")
                    content = " ".join([p.get_text(strip=True) for p in content_paras])

                    time_tag = art_soup.select_one("div.read__time")
                    published = extract_datetime_from_title(time_tag.get_text(strip=True) if time_tag else "", url)

                    if not published:
                        jakarta_tz = pytz.timezone("Asia/Jakarta")
                        published = datetime.now(jakarta_tz).strftime("%Y-%m-%d %H:%M")

                    if is_relevant(title, query, content):
                        data.append({
                            "source": get_source_from_url(url),
                            "title": title,
                            "description": "",
                            "content": content,
                            "url": url,
                            "publishedAt": published
                        })
                except Exception as e:
                    continue
                if len(data) >= max_articles:
                    break
            return pd.DataFrame(data)
        else:
            st.warning(f"Gagal scraping Kompas: Status code {res.status_code}.")
    except (requests.exceptions.RequestException, Exception) as e:
        st.warning(f"Gagal scraping Kompas: {e}.")
    return pd.DataFrame()

@st.cache_data(show_spinner="Menggabungkan hasil...")
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
        df = pd.concat(dfs, ignore_index=True)
        return df
    else:
        return pd.DataFrame()

# --- FUNGSI UNTUK GITHUB API ---
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
        # Tambahkan baris ini untuk memastikan DataFrame memiliki kolom-kolom yang diperlukan
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

    tz = pytz.timezone("Asia/Jakarta")
    now = datetime.now(tz).strftime("%A, %d %B %Y %H:%M")

    for _, row in all_articles.iterrows():
        article_log = {
            "user_id": user_id,
            "query": query,
            "title": str(row.get('title', '')),
            "url": str(row.get('url', '')),
            "content": str(row.get('content', '')),
            "source": str(row.get('source', '')),
            "click_time": now,
            "publishedAt": row.get('publishedAt', now),
            "label": 1 if row.get('url', '') in clicked_urls else 0
        }
        history_list.append(article_log)

    updated_content = json.dumps(history_list, indent=2, ensure_ascii=False)
    repo.update_file(
        st.secrets["file_path"],
        f"Update history for {query}",
        updated_content,
        contents.sha
    )

def get_recent_queries_by_days(user_id, df, days=3):
    if df.empty or "user_id" not in df.columns or 'click_time' not in df.columns:
        return {}

    df_user = df[df["user_id"] == user_id].copy()
    jakarta_tz = pytz.timezone("Asia/Jakarta")

    if 'publishedAt' not in df_user.columns:
        df_user['publishedAt'] = df_user['click_time']

    df_user['date_to_process'] = df_user['click_time']

    df_user["timestamp"] = pd.to_datetime(
        df_user["date_to_process"],
        format="%A, %d %B %Y %H:%M",
        errors='coerce'
    )
    df_user["timestamp"].fillna(
        pd.to_datetime(
            df_user["date_to_process"],
            format="%Y-%m-%d %H:%M",
            errors='coerce'
        ), inplace=True
    )
    df_user = df_user.dropna(subset=['timestamp']).copy()

    try:
        df_user['timestamp'] = pd.to_datetime(df_user['timestamp']).dt.tz_localize(jakarta_tz, ambiguous='NaT', nonexistent='NaT')
    except Exception:
        return {}

    df_user = df_user.dropna(subset=['timestamp']).copy()
    if df_user.empty:
        return {}

    now = datetime.now(jakarta_tz)
    cutoff_time = now - timedelta(days=days)
    recent_df = df_user[df_user["timestamp"] >= cutoff_time].copy()

    if recent_df.empty:
        return {}

    recent_df.loc[:, 'date'] = recent_df['timestamp'].dt.strftime('%d %B %Y')
    grouped_queries = recent_df.groupby('date')['query'].unique().to_dict()

    sorted_dates = sorted(
        grouped_queries.keys(),
        key=lambda d: datetime.strptime(d, '%d %B %Y'),
        reverse=True
    )
    ordered_grouped_queries = {date: grouped_queries[date] for date in sorted_dates}
    return ordered_grouped_queries

def get_most_frequent_topics(user_id, df, days=3):
    if df.empty or "user_id" not in df.columns:
        return []
    df_user = df[df["user_id"] == user_id].copy()
    jakarta_tz = pytz.timezone("Asia/Jakarta")

    if 'publishedAt' not in df_user.columns:
        df_user['publishedAt'] = df_user['click_time']

    df_user['date_to_process'] = df_user['publishedAt']

    df_user["timestamp"] = pd.to_datetime(
        df_user["date_to_process"],
        format="%Y-%m-%d %H:%M",
        errors='coerce'
    )
    df_user["timestamp"].fillna(
        pd.to_datetime(
            df_user["date_to_process"],
            format="%A, %d %B %Y %H:%M",
            errors='coerce'
        ), inplace=True
    )
    df_user = df_user.dropna(subset=['timestamp'])

    now = datetime.now(jakarta_tz)
    cutoff_time = now - timedelta(days=days)

    recent_df = df_user[df_user["timestamp"] >= cutoff_time]

    if recent_df.empty:
        return []

    query_counts = Counter(recent_df['query'])
    sorted_queries = sorted(query_counts.items(), key=lambda item: item[1], reverse=True)
    return sorted_queries

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
        title = str(row.get("title", ""))
        content = str(row.get("content", ""))
        text = preprocess_text(title + " " + content)
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
        st.warning("⚠️ Gagal melatih model: hanya ada satu jenis label (perlu klik & tidak klik).")
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

def recommend(df, query, clf, n_per_source=3, min_score=0.5):
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.drop_duplicates(subset=['url'], inplace=True)

    df["processed"] = df.apply(lambda row: preprocess_text(row['title'] + ' ' + row.get('content', '')), axis=1)
    vec = model_sbert.encode(df["processed"].tolist())

    df['publishedAt_dt'] = pd.to_datetime(df['publishedAt'], errors='coerce')
    df['vec_temp'] = list(vec)
    df = df.dropna(subset=['publishedAt_dt'])
    vec = df['vec_temp'].tolist()
    df = df.drop(columns=['vec_temp'])

    if df.empty:
        return pd.DataFrame()

    if clf:
        scores = clf.predict_proba(vec)[:, 1]
        df["score"] = scores
        df["bonus"] = df["title"].apply(lambda x: 0.1 if query.lower() in x.lower() else 0)
        df["final_score"] = (df["score"] + df["bonus"]).clip(0, 1)

        df = df[df['final_score'] >= min_score].copy()

        def top_n(x):
            return x.sort_values(by=['publishedAt_dt', 'final_score'], ascending=[False, False]).head(n_per_source)
        top_n_per_source = df.groupby("source", group_keys=False).apply(top_n, include_groups=False)
        return top_n_per_source.sort_values(by=['publishedAt_dt', 'final_score'], ascending=[False, False]).reset_index(drop=True)
    else:
        q_vec = model_sbert.encode([preprocess_text(query)])
        sims = cosine_similarity(q_vec, vec)[0]
        df["similarity"] = sims

        df = df[df['similarity'] >= min_score].copy()

        def top_n_sim(x):
            return x.sort_values(by=['publishedAt_dt', 'similarity'], ascending=[False, False]).head(n_per_source)
        top_n_per_source = df.groupby("source", group_keys=False).apply(top_n_sim, include_groups=False)
        return top_n_per_source.sort_values(by=['publishedAt_dt', 'similarity'], ascending=[False, False]).reset_index(drop=True)

def handle_js_click(url):
    if url not in st.session_state.clicked_urls_in_session:
        st.session_state.clicked_urls_in_session.append(url)
        st.rerun()

def main():
    st.title("📰 Sistem Rekomendasi Berita")
    st.markdown("Aplikasi ini merekomendasikan berita dari Detik, CNN, dan Kompas berdasarkan riwayat pencarian Anda.")

    if st.sidebar.button("Bersihkan Cache & Muat Ulang"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cache berhasil dibersihkan! Aplikasi akan dimuat ulang.")
        time.sleep(1)
        st.rerun()

    if st.session_state.history.empty:
        st.session_state.history = load_history_from_github()
    
    st.sidebar.header("Model Personalisasi")
    df_train = build_training_data(USER_ID)
    clf = None
    if not df_train.empty and df_train['label'].nunique() > 1:
        st.sidebar.success("Model berhasil dilatih.")
        clf = train_model(df_train)
    else:
        st.sidebar.info("Model belum bisa dilatih karena riwayat tidak mencukupi. Silakan lakukan pencarian dan klik tautan artikel.")

    st.header("📚 Pencarian Berita per Tanggal")
    grouped_queries = get_recent_queries_by_days(USER_ID, st.session_state.history, days=3)

    if grouped_queries:
        for date, queries in grouped_queries.items():
            st.subheader(f"Tanggal {date}")
            unique_queries = sorted(list(set(queries)))
            for q in unique_queries:
                with st.expander(f"- {q}"):
                    df_filtered = st.session_state.history[(st.session_state.history['query'] == q) & (st.session_state.history['user_id'] == USER_ID)].copy()
                    
                    if df_filtered.empty:
                        st.info("❗ Tidak ditemukan berita dalam riwayat untuk topik ini.")
                        continue
                    
                    if 'publishedAt' not in df_filtered.columns:
                        df_filtered['publishedAt'] = df_filtered['click_time']

                    df_filtered['publishedAt_dt'] = pd.to_datetime(df_filtered['publishedAt'], errors='coerce')
                    df_filtered.dropna(subset=['publishedAt_dt'], inplace=True)
                    
                    if df_filtered.empty:
                        st.info("❗ Setelah pembersihan data, tidak ada entri yang valid.")
                        continue

                    df_filtered['processed'] = df_filtered.apply(lambda row: preprocess_text(row['title'] + ' ' + str(row.get('content', ''))), axis=1)
                    
                    q_vec = model_sbert.encode([preprocess_text(q)])
                    df_filtered['similarity'] = df_filtered['processed'].apply(lambda x: cosine_similarity([model_sbert.encode(x)], q_vec)[0][0])
                    
                    if clf:
                        if 'final_score' not in df_filtered.columns:
                            df_filtered['final_score'] = clf.predict_proba(model_sbert.encode(df_filtered['processed'].tolist()))[:, 1]
                        
                        def top_n_history(x):
                            return x.sort_values(by=['publishedAt_dt', 'final_score'], ascending=[False, False]).head(3)
                        articles_to_show = df_filtered.groupby("source", group_keys=False).apply(top_n_history, include_groups=False)
                        articles_to_show = articles_to_show.sort_values(by=['publishedAt_dt', 'final_score'], ascending=[False, False]).reset_index(drop=True)
                        skor_key = 'final_score'
                    else:
                        def top_n_history(x):
                            return x.sort_values(by=['publishedAt_dt', 'similarity'], ascending=[False, False]).head(3)
                        articles_to_show = df_filtered.groupby("source", group_keys=False).apply(top_n_history, include_groups=False)
                        articles_to_show = articles_to_show.sort_values(by=['publishedAt_dt', 'similarity'], ascending=[False, False]).reset_index(drop=True)
                        skor_key = 'similarity'

                    if articles_to_show.empty:
                            st.info("❗ Tidak ada hasil relevan yang ditemukan dalam riwayat untuk topik ini.")
                            continue
                    
                    for i, row in articles_to_show.iterrows():
                        source_name = get_source_from_url(row['url'])
                        
                        display_time = row.get('publishedAt', 'Tidak Diketahui')
                        try:
                            dt_obj = datetime.strptime(display_time, "%Y-%m-%d %H:%M")
                            if dt_obj.strftime("%H:%M") == "00:00":
                                formatted_time = dt_obj.strftime("%Y-%m-%d")
                            else:
                                formatted_time = dt_obj.strftime("%Y-%m-%d %H:%M")
                        except (ValueError, TypeError):
                            formatted_time = display_time

                        st.markdown(f"**[{source_name}]** {row['title']}")
                        st.markdown(f"[{row['url']}]({row['url']})")
                        st.write(f"Waktu Publikasi: *{formatted_time}*")
                        if skor_key in row:
                            st.write(f"Skor: `{row[skor_key]:.2f}`")
                        st.markdown("---")
    else:
        st.info("Belum ada riwayat pencarian pada 3 hari terakhir.")

    st.markdown("---")

    st.header("🔥 Rekomendasi Berita Hari Ini")
    most_frequent_topics = get_most_frequent_topics(USER_ID, st.session_state.history, days=3)
    if most_frequent_topics:
        q, count = most_frequent_topics[0]
        with st.spinner('Mencari berita...'):
            df_news = scrape_all_sources(q)
        if df_news.empty:
            st.info("❗ Tidak ditemukan berita.")
        else:
            results = recommend(df_news, q, clf, n_per_source=1, min_score=0.5)
            if results.empty:
                st.info("❗ Tidak ada hasil relevan.")
            else:
                for i, row in results.iterrows():
                    source_name = get_source_from_url(row['url'])
                    st.markdown(f"**[{source_name}]** {row['title']}")
                    st.markdown(row['url'])
                    
                    display_time = row.get('publishedAt', 'Tidak Diketahui')
                    try:
                        dt_obj = datetime.strptime(display_time, "%Y-%m-%d %H:%M")
                        if dt_obj.strftime("%H:%M") == "00:00":
                            formatted_time = dt_obj.strftime("%Y-%m-%d")
                        else:
                            formatted_time = dt_obj.strftime("%Y-%m-%d %H:%M")
                    except (ValueError, TypeError):
                        formatted_time = display_time

                    st.write(f"Waktu: *{formatted_time}*")
                    skor_key = 'final_score' if 'final_score' in row else 'similarity'
                    st.write(f"Skor: `{row[skor_key]:.2f}`")

                    st.markdown("---")
    else:
        st.info("🔥 Tidak ada topik yang sering dicari dalam 3 hari terakhir.")

    st.markdown("---")

    st.header("🔍 Pencarian Berita")
    search_query = st.text_input("Ketik topik berita yang ingin Anda cari:", key="search_input")

    if st.button("Cari Berita"):
        if search_query:
            if 'current_query' in st.session_state and st.session_state.current_query:
                save_interaction_to_github(USER_ID, st.session_state.current_query, st.session_state.current_recommended_results, st.session_state.clicked_urls_in_session)
                st.cache_data.clear()
                st.session_state.history = load_history_from_github()

            with st.spinner('Mengambil berita dan merekomendasikan...'):
                st.session_state.current_search_results = scrape_all_sources(search_query)
                results = recommend(st.session_state.current_search_results, search_query, clf, n_per_source=3, min_score=0.5)
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
            for i, row in st.session_state.current_recommended_results.iterrows():
                source_name = get_source_from_url(row['url'])
                
                button_html = f"""<style>.styled-button {{ background-color: #007bff; color: white; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 8px; border: none; }}</style><button class="styled-button" onclick="window.parent.postMessage({{ streamlit: true, event: 'st_event', data: {{ url: '{row['url']}' }} }}, '*'); window.open('{row['url']}', '_blank');">Buka Artikel & Catat Interaksi</button>"""

                st.markdown(f"**[{source_name}]** {row['title']}")
                st.markdown(f"[{row['url']}]({row['url']})")
                
                display_time = row.get('publishedAt', 'Tidak Diketahui')
                try:
                    dt_obj = datetime.strptime(display_time, "%Y-%m-%d %H:%M")
                    if dt_obj.strftime("%H:%M") == "00:00":
                        formatted_time = dt_obj.strftime("%Y-%m-%d")
                    else:
                        formatted_time = dt_obj.strftime("%Y-%m-%d %H:%M")
                except (ValueError, TypeError):
                    formatted_time = display_time

                st.write(f"Waktu: *{formatted_time}*")
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
