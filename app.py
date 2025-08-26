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

# --- KONFIGURASI HALAMAN STREAMLIT ---
st.set_page_config(page_title="Sistem Rekomendasi Berita", layout="wide")

# --- Konfigurasi dan Inisialisasi ---
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
        # fallback lebih kecil jika model utama gagal diunduh
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

# 3. Ekstrak waktu
@st.cache_data
def extract_datetime_from_title(title):
    bulan_mapping = {
        "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04", "Mei": "05", "Jun": "06",
        "Jul": "07", "Agu": "08", "Ags": "08", "Agustus": "08", "Sep": "09",
        "September": "09", "Okt": "10", "Oktober": "10", "Nov": "11",
        "November": "11", "Des": "12", "Desember": "12", "Januari": "01",
        "Februari": "02", "Maret": "03", "April": "04", "Juni": "06", "Juli": "07"
    }
    zona = pytz.timezone("Asia/Jakarta")

    # Gaya Kompas: "Kompas.com - 25/08/2025, 14:12"
    pattern_kompas = r"Kompas\.com\s*-\s*(\d{2})/(\d{2})/(\d{4}),\s*(\d{2}:\d{2})"
    match = re.search(pattern_kompas, title)
    if match:
        day, month, year, time_str = match.groups()
        try:
            dt = datetime.strptime(f"{year}-{month}-{day} {time_str}", "%Y-%m-%d %H:%M")
            return zona.localize(dt).strftime("%Y-%m-%d %H:%M")
        except:
            pass

    # "Senin, 25 Agustus 2025 14:12" atau tanpa nama hari
    pattern1 = r"(?:\w+, )?(\d{1,2}) (\w+) (\d{4}) (\d{2}:\d{2})"
    match = re.search(pattern1, title)
    if match:
        day, month_str, year, time_str = match.groups()
        month = bulan_mapping.get(month_str)
        if month:
            try:
                dt = datetime.strptime(f"{year}-{month}-{int(day):02d} {time_str}", "%Y-%m-%d %H:%M")
                return zona.localize(dt).strftime("%Y-%m-%d %H:%M")
            except:
                pass

    # "25 Agustus 2025"
    pattern2 = r"(\d{1,2}) (\w+) (\d{4})"
    match = re.search(pattern2, title)
    if match:
        day, month_str, year = match.groups()
        month = bulan_mapping.get(month_str)
        if month:
            try:
                dt = datetime.strptime(f"{year}-{month}-{int(day):02d}", "%Y-%m-%d")
                return zona.localize(dt).strftime("%Y-%m-%d %H:%M")
            except:
                pass

    # "25/08/2025, 14:12"
    pattern3 = r"(\d{2})/(\d{2})/(\d{4}), (\d{2}:\d{2})"
    match = re.search(pattern3, title)
    if match:
        day, month, year, time_str = match.groups()
        try:
            dt = datetime.strptime(f"{year}-{month}-{day} {time_str}", "%Y-%m-%d %H:%M")
            return zona.localize(dt).strftime("%Y-%m-%d %H:%M")
        except:
            pass

    # "5 menit yang lalu" / "2 jam yang lalu"
    pattern4 = r"(\d+)\s+(menit|jam)\s+yang lalu"
    match = re.search(pattern4, title)
    if match:
        jumlah, satuan = match.groups()
        try:
            delta = timedelta(minutes=int(jumlah)) if satuan == "menit" else timedelta(hours=int(jumlah))
            dt = datetime.now(zona) - delta
            return dt.strftime("%Y-%m-%d %H:%M")
        except:
            pass

    return ""

@st.cache_data
def is_relevant(title, query, content="", threshold=0.35):
    combined = f"{title} {content}"
    combined_vecs = model_sbert.encode([combined])
    query_vecs = model_sbert.encode([query])
    sim = cosine_similarity(combined_vecs, query_vecs)[0][0]
    return sim >= threshold

# 4. Scraper per sumber (DENGAN PENAMBAHAN RETRY DAN FALLBACK TANGGAL)
@st.cache_data(show_spinner="Mencari berita di Detik...")
def scrape_detik(query, max_articles=15):
    url = f"https://www.detik.com/search/searchall?query={query.replace(' ', '+')}"
    data = []
    for _ in range(2):
        try:
            res = requests.get(url, headers=HEADERS, timeout=10)
            if res.status_code == 200:
                soup = BeautifulSoup(res.content, "html.parser")
                articles_raw = soup.select("article.list-content__item")
                for article in articles_raw:
                    try:
                        title_tag = article.find('h3', class_='media__title')
                        link = title_tag.a['href'] if title_tag and title_tag.a else ''
                        description_tag = article.find('div', class_='media__desc')
                        date_tag = article.find('div', class_='media__date').find('span') if article.find('div', class_='media__date') else None

                        if not title_tag or not link:
                            continue

                        title = title_tag.get_text(strip=True)
                        description = description_tag.get_text(strip=True) if description_tag else ""
                        published_at = date_tag.get('title', '') if date_tag else ''
                        published_at = extract_datetime_from_title(published_at)

                        if not published_at:
                            jakarta_tz = pytz.timezone("Asia/Jakarta")
                            published_at = datetime.now(jakarta_tz).strftime("%Y-%m-%d %H:%M")

                        if is_relevant(title, query, description):
                            data.append({
                                "source": get_source_from_url(link),
                                "title": title,
                                "description": description,
                                "content": f"{title} {description}",
                                "url": link,
                                "publishedAt": published_at
                            })
                    except Exception:
                        continue
                    if len(data) >= max_articles:
                        break
                return pd.DataFrame(data)
            else:
                time.sleep(2)
        except (requests.exceptions.RequestException, Exception):
            time.sleep(2)
    return pd.DataFrame()

@st.cache_data(show_spinner="Mencari berita di CNN...")
def scrape_cnn_fixed(query, max_results=10):
    feed_urls = [
        "https://www.cnnindonesia.com/nasional/rss",
        "https://www.cnnindonesia.com/internasional/rss",
        "https://www.cnnindonesia.com/ekonomi/rss",
    ]
    results = []
    for feed_url in feed_urls:
        try:
            feed = feedparser.parse(feed_url)
            if feed.entries:
                for entry in feed.entries:
                    title = entry.title.strip()
                    link = entry.link
                    summary = getattr(entry, "summary", "").strip()
                    published = getattr(entry, "published", "")

                    published_at = ""
                    try:
                        dt = datetime.strptime(published, "%a, %d %b %Y %H:%M:%S %z").astimezone(pytz.timezone("Asia/Jakarta"))
                        published_at = dt.strftime("%Y-%m-%d %H:%M")
                    except:
                        jakarta_tz = pytz.timezone("Asia/Jakarta")
                        published_at = datetime.now(jakarta_tz).strftime("%Y-%m-%d %H:%M")

                    if is_relevant(title, query, summary):
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
        except Exception:
            continue
    return pd.DataFrame(results)

@st.cache_data(show_spinner="Mencari berita di Kompas...")
def scrape_kompas_fixed(query, max_articles=10):
    search_url = f"https://search.kompas.com/search?q={query.replace(' ', '+')}"
    data = []
    for _ in range(2):
        try:
            res = requests.get(search_url, headers=HEADERS, timeout=10)
            if res.status_code == 200:
                soup = BeautifulSoup(res.text, "html.parser")
                articles = soup.select("div.articleItem")[:max_articles]
                if not articles:
                    return pd.DataFrame()
                for item in articles:
                    try:
                        a_tag = item.select_one("a.article-link")
                        title_tag = item.select_one("h2.articleTitle")
                        if not a_tag or not title_tag:
                            continue
                        url = a_tag["href"]
                        title = title_tag.get_text(strip=True)

                        time.sleep(random.uniform(1, 2))
                        art_res = requests.get(url, headers=HEADERS, timeout=10)
                        art_soup = BeautifulSoup(art_res.text, "html.parser")
                        content_paras = art_soup.select("div.read__content > p")
                        content = " ".join([p.get_text(strip=True) for p in content_paras])

                        time_tag = art_soup.select_one("div.read__time")
                        published = extract_datetime_from_title(time_tag.get_text(strip=True)) if time_tag else ""

                        if (not published) or published.endswith("00:00"):
                            url_match = re.search(r"/(\d{4})/(\d{2})/(\d{2})/(\d{2})(\d{2})", url)
                            if url_match:
                                y, m, d, h, mi = url_match.groups()
                                dt = datetime.strptime(f"{y}-{m}-{d} {h}:{mi}", "%Y-%m-%d %H:%M")
                                dt = pytz.timezone("Asia/Jakarta").localize(dt)
                                published = dt.strftime("%Y-%m-%d %H:%M")

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
                    except Exception:
                        continue
                    if len(data) >= max_articles:
                        break
                return pd.DataFrame(data)
            else:
                time.sleep(2)
        except (requests.exceptions.RequestException, Exception):
            time.sleep(2)
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
        return pd.DataFrame(data) if isinstance(data, list) and data else pd.DataFrame()
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
            "click_time": now,
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
    """Mengembalikan dict terurut: { '25 Agustus 2025': ['gempa','pemilu',...], ... }"""
    if df.empty or "click_time" not in df.columns:
        return {}

    df_user = df[df["user_id"] == user_id].copy()
    jakarta_tz = pytz.timezone("Asia/Jakarta")

    # Konversi kolom 'click_time' ke datetime, dengan NaT untuk yang gagal.
    df_user["timestamp"] = pd.to_datetime(
        df_user["click_time"],
        format="%A, %d %B %Y %H:%M",
        errors='coerce'
    )

    # MEMBUANG BARIS DENGAN NILAI NaT (NOT A TIME)
    # Ini adalah perbaikan utama untuk AttributeError.
    df_user = df_user.dropna(subset=['timestamp']).copy()
    
    # Jika setelah filtering DataFrame menjadi kosong, kembalikan dict kosong
    if df_user.empty:
        return {}
        
    # Mengatur zona waktu secara eksplisit
    if df_user["timestamp"].dt.tz is None:
        df_user.loc[:, "timestamp"] = df_user["timestamp"].dt.tz_localize(jakarta_tz)
    else:
        df_user.loc[:, "timestamp"] = df_user["timestamp"].dt.tz_convert(jakarta_tz)
    
    now = datetime.now(jakarta_tz)
    cutoff_time = now - timedelta(days=days)
    recent_df = df_user[df_user["timestamp"] >= cutoff_time].copy()
    
    if recent_df.empty:
        return {}
    
    # Sekarang kolom 'timestamp' sudah pasti bertipe datetime, sehingga .dt aman digunakan.
    recent_df.loc[:, 'date'] = recent_df['timestamp'].dt.strftime('%d %B %Y')
    grouped_queries = recent_df.groupby('date')['query'].unique().to_dict()

    sorted_dates = sorted(
        grouped_queries.keys(),
        key=lambda d: datetime.strptime(d, '%d %B %Y'),
        reverse=True
    )
    ordered_grouped_queries = {date: grouped_queries[date] for date in sorted_dates}
    return ordered_grouped_queries

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

def recommend(df, query, clf, n_per_source=3):
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

        def top_n(x):
            return x.sort_values(by=['publishedAt_dt', 'final_score'], ascending=[False, False]).head(n_per_source)
        top_n_per_source = df.groupby("source", group_keys=False).apply(top_n)
        return top_n_per_source.sort_values(by=['publishedAt_dt', 'final_score'], ascending=[False, False]).reset_index(drop=True)
    else:
        q_vec = model_sbert.encode([preprocess_text(query)])
        sims = cosine_similarity(q_vec, vec)[0]
        df["similarity"] = sims

        def top_n_sim(x):
            return x.sort_values(by=['publishedAt_dt', 'similarity'], ascending=[False, False]).head(n_per_source)
        top_n_per_source = df.groupby("source", group_keys=False).apply(top_n_sim)
        return top_n_per_source.sort_values(by=['publishedAt_dt', 'similarity'], ascending=[False, False]).reset_index(drop=True)

def main():
    st.title("📰 Sistem Rekomendasi Berita")
    st.markdown("Aplikasi ini merekomendasikan berita dari Detik, CNN, dan Kompas berdasarkan riwayat pencarian Anda.")

    if st.sidebar.button("Bersihkan Cache & Muat Ulang"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cache berhasil dibersihkan! Aplikasi akan dimuat ulang.")
        time.sleep(1)
        st.rerun()

    if 'history' not in st.session_state:
        st.session_state.history = load_history_from_github()
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

    st.sidebar.header("Model Personalisasi")
    df_train = build_training_data(USER_ID)
    clf = None
    if not df_train.empty and df_train['label'].nunique() > 1:
        st.sidebar.success("Model berhasil dilatih.")
        clf = train_model(df_train)
    else:
        st.sidebar.info("Model belum bisa dilatih karena riwayat tidak mencukupi. Silakan lakukan pencarian dan klik tautan artikel.")

    # --- PENCARIAN PER TANGGAL ---
    st.header("📚 Pencarian Berita per Tanggal")
    grouped_queries = get_recent_queries_by_days(USER_ID, st.session_state.history, days=3)

    if grouped_queries:
        for date, queries in grouped_queries.items():
            st.subheader(f"Tanggal {date}")
            unique_queries = sorted(list(set(queries)))
            for q in unique_queries:
                with st.expander(f"- {q}"):
                    with st.spinner('Mencari berita...'):
                        df_news = scrape_all_sources(q)
                    if df_news.empty:
                        st.info("❗ Tidak ditemukan berita.")
                        continue

                    results = recommend(df_news, q, clf, n_per_source=3)
                    if results.empty:
                        st.info("❗ Tidak ada hasil relevan.")
                        continue
                    else:
                        for i, row in results.iterrows():
                            source_name = get_source_from_url(row['url'])
                            if 'url' in row and row['url']:
                                st.markdown(f"**[{source_name}]** {row['title']}")
                                st.markdown(f"[{row['url']}]({row['url']})")
                            else:
                                st.markdown(f"**[{source_name}]** {row['title']}")
                                st.info("Tautan tidak tersedia.")
                            st.write(f"Waktu: *{row['publishedAt']}*")
                            skor_key = 'final_score' if 'final_score' in row else 'similarity'
                            st.write(f"Skor: `{row[skor_key]:.2f}`")

                            key_link = f"link_{i}_{row.get('url', 'no_url')}"
                            if st.button(f"Catat Interaksi", key=key_link):
                                st.session_state.clicked_urls_in_session.append(row['url'])
                                st.toast("Interaksi Anda telah dicatat untuk sesi ini.")
                            st.markdown("---")

            if st.session_state.current_query:
                st.info(f"Anda telah mencatat {len(st.session_state.clicked_urls_in_session)} artikel. Data akan disimpan saat Anda memulai pencarian baru.")
    else:
        st.info("Belum ada riwayat pencarian pada 3 hari terakhir.")


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

if __name__ == "__main__":
    main()
