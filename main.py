from googleapiclient.discovery import build
import pandas as pd
from datetime import date, timedelta
import time
from isodate import parse_duration
from googleapiclient.errors import HttpError
import re
from transformers import pipeline, AutoTokenizer
from langdetect import detect, LangDetectException
import emoji
import psycopg2 as pg
import analysis

game_name = input("Wpisz nazwe gry: ").lower()
time_frame = input("Wpisz przedzial czasowy (days): ").lower()
start_time = time.time()
### API YT
API_KEY = "AIz"


def fetch_yt_data(game_name, video_duration):
    """
    Fetches YouTube video data for a given game name.
    Returns a DataFrame with video details.
    """
    youtube = build("youtube", "v3", developerKey="AIz")

    filter_date = date.today() - timedelta(days=int(time_frame))
    next_page_token = None
    old_video = True
    df = pd.DataFrame()
    df["comments"] = ""

    ############################### pobieranie danych z YT

    while old_video:
        requests = youtube.search().list(
            part="snippet",
            q=game_name,
            type="video",
            order="date",
            videoDuration=video_duration,
            eventType="completed",
            pageToken=next_page_token,
            maxResults=50,
        )
        vid_ids = []
        response = requests.execute()
        for item in response["items"]:
            if item["snippet"]["publishedAt"] >= filter_date.isoformat():
                video_id = item["id"]["videoId"]
                vid_ids.append(video_id)
                df.loc[video_id, "title"] = item["snippet"]["title"]
                df.loc[video_id, "publishedAt"] = item["snippet"]["publishedAt"].split(
                    "T"
                )[0]
                df.loc[video_id, "channelName"] = item["snippet"]["channelTitle"]
            else:
                old_video = False
                break

        requests = youtube.videos().list(
            part="statistics,contentDetails", id=",".join(vid_ids)
        )

        vidresponse = requests.execute()
        for item in vidresponse["items"]:
            video_id = item["id"]
            if video_id in df.index:
                df.loc[video_id, "duration"] = parse_duration(
                    item["contentDetails"]["duration"]
                ).total_seconds()
                df.loc[video_id, "viewCount"] = int(
                    item["statistics"].get("viewCount", 0)
                )
                df.loc[video_id, "likeCount"] = int(
                    item["statistics"].get("likeCount", 0)
                )
                df.loc[video_id, "commentCount"] = int(
                    item["statistics"].get("commentCount", 0)
                )

            else:
                print(f"No stats for ID: {video_id}")

        for video_id in vid_ids:
            comments_next_page_token = None
            comments_list = []
            try:
                while True:
                    comment_response = youtube.commentThreads().list(
                        part="snippet",
                        videoId=video_id,
                        textFormat="plainText",
                        pageToken=comments_next_page_token,
                        maxResults=100,
                    )

                    requests = comment_response.execute()
                    for item in requests["items"]:
                        comment_snippet = item["snippet"]["topLevelComment"]["snippet"]
                        comments_list.append(comment_snippet["textDisplay"])

                    comments_next_page_token = requests.get("nextPageToken", None)

                    if len(comments_list) >= 200:
                        df.at[video_id, "comments"] = comments_list
                        break

                    if not comments_next_page_token:
                        df.at[video_id, "comments"] = comments_list
                        break

            except HttpError as error:
                if error.resp.status == 403:
                    print(f"Error 403: {video_id}")
                    df.at[video_id, "comments"] = []

        next_page_token = response.get("nextPageToken")
        time.sleep(1)

        if not next_page_token:
            break

    next_page_token = None
    return df


def check_if_video_exists(cursor, index):
    query = f""" SELECT index FROM {game_name_db}_data WHERE index = %s"""
    cursor.execute(query, (index,))
    return cursor.fetchone() is not None


def update_video_data(
    cursor,
    index,
    viewCount,
    likeCount,
    commentCount,
    comments,
    lang,
    translated_comments,
    sentiment,
    sentiment_summary,
):
    query = f""" UPDATE {game_name_db}_data 
                  SET viewCount = %s, likeCount = %s, commentCount = %s, comments = %s, lang = %s, translated_comments = %s, sentiment = %s, sentiment_summary = %s, refresh_date = CURRENT_DATE
                  WHERE index = %s"""
    cursor.execute(
        query,
        (
            viewCount,
            likeCount,
            commentCount,
            comments,
            lang,
            translated_comments,
            sentiment,
            sentiment_summary,
            index,
        ),
    )


def insert_video_data(
    cursor,
    index,
    title,
    publishedAt,
    channelName,
    duration,
    viewCount,
    likeCount,
    commentCount,
    comments,
    lang,
    translated_comments,
    sentiment,
    sentiment_summary,
):
    query = f""" INSERT INTO {game_name_db}_data (index, title, publishedAt, channelName, duration, viewCount, likeCount, commentCount, comments, lang, translated_comments, sentiment, sentiment_summary, refresh_date)
                  VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_DATE)"""
    cursor.execute(
        query,
        (
            index,
            title,
            publishedAt,
            channelName,
            duration,
            viewCount,
            likeCount,
            commentCount,
            comments,
            lang,
            translated_comments,
            sentiment,
            sentiment_summary,
        ),
    )


df = pd.concat([fetch_yt_data(game_name, "medium"), fetch_yt_data(game_name, "long")])
df = df[df.columns[1:].tolist() + [df.columns[0]]]
df.to_csv("youtube_data.csv")
# df2 = pd.read_csv('youtube_data.csv', index_col=0)

# Analiza sentymentu komentarzy
df_exploded = df[df["comments"].str.len() > 0]
df_exploded = df_exploded.explode("comments").reset_index()


def normalize_comment(comment):

    comment = comment.lower()
    comment = comment.replace("\n", " ").strip()
    comment = re.sub(r"http\S+", "", comment)
    comment = re.sub(r"@\w+", "", comment)  # Usunięcie wzmiankowania
    comment = re.sub(r"#[\w-]+", "", comment)  # Usunięcie hashtagów

    comment = emoji.demojize(comment)  # zamienia emoji na tekst, np. :red_heart:
    comment = re.sub(r":\w+:", "", comment)
    comment = re.sub(r"[^\w\s.,?!-]", "", comment, flags=re.UNICODE)

    timestamp_pattern = r"\b(\d{1,2}:\d{2}(:\d{2})?)\b"
    comment = re.sub(timestamp_pattern, "", comment)
    return comment


def safe_detect(text):
    try:
        if text and len(text.strip()) > 3:
            return detect(text)
        else:
            return "unknown"
    except LangDetectException:
        return "unknown"


tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
translation_pipeline = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-mul-en",
    max_length=512,
    truncation=True,
    tokenizer=tokenizer,
)

df_exploded["comments"] = df_exploded["comments"].astype(str).apply(normalize_comment)
print("starting lang detection")
df_exploded["lang"] = df_exploded["comments"].apply(safe_detect)

to_translate = df_exploded[
    (df_exploded["lang"] != "en") & (df_exploded["lang"] != "unknown")
]["comments"].tolist()
eng_not_translated = df_exploded[df_exploded["lang"] == "en"]["comments"].tolist()
unknwon_not_translated = df_exploded[df_exploded["lang"] == "unknown"][
    "comments"
].tolist()
print("starting translation")

batch_size = 50
translated_texts = []
for i in range(0, len(to_translate), batch_size):
    batch = to_translate[i : i + batch_size]
    batch_results = translation_pipeline(batch)
    translated_texts.extend([res["translation_text"] for res in batch_results])
    print(f"translated: {i + len(batch)} / {len(to_translate)}")


df_exploded["translated_comments"] = ""

# rozdzielic to na 3 grupy zeby pozniej nie robic sentymentu z unknown   !!!!!!

df_exploded.loc[
    (df_exploded["lang"] != "en") & (df_exploded["lang"] != "unknown"),
    "translated_comments",
] = translated_texts
df_exploded.loc[(df_exploded["lang"] == "en"), "translated_comments"] = (
    eng_not_translated
)
df_exploded.loc[(df_exploded["lang"] == "unknown"), "translated_comments"] = (
    unknwon_not_translated
)

# analiza sentymentu
print("starting sentiment analysis")
sentiment_pipeline = pipeline("sentiment-analysis")
comments_to_analyze = df_exploded[(df_exploded["lang"] != "unknown")][
    "translated_comments"
].tolist()

sentiment_results_list = []
batch_size_sentiment = 100  # Wielkość paczki dla analizy sentymentu

for i in range(0, len(comments_to_analyze), batch_size_sentiment):
    batch_sentiment = comments_to_analyze[i : i + batch_size_sentiment]
    batch_results_sentiment = sentiment_pipeline(
        batch_sentiment, max_length=512, truncation=True
    )
    sentiment_results_list.extend(batch_results_sentiment)
    print(f"sentiment: {i + len(batch_sentiment)} / {len(comments_to_analyze)}")

sentiments = [
    1 if result["label"] == "POSITIVE" else -1 if result["label"] == "NEGATIVE" else 0
    for result in sentiment_results_list
]

df_exploded["sentiment"] = 0
df_exploded.loc[df_exploded["lang"] != "unknown", "sentiment"] = sentiments


df_final = (
    df_exploded.groupby("index")
    .agg(
        lang=("lang", list),
        translated_comments=("translated_comments", list),
        sentiment=("sentiment", list),
    )
    .reset_index()
)

df = df.merge(df_final, left_index=True, right_on="index", how="left")

# cleanup

df["translated_comments"] = (
    df["translated_comments"]
    .fillna("")
    .apply(lambda x: x if isinstance(x, list) else [])
)
df["sentiment"] = (
    df["sentiment"].fillna("").apply(lambda x: x if isinstance(x, list) else [])
)
df["lang"] = df["lang"].fillna("").apply(lambda x: x if isinstance(x, list) else [])

df.loc[df["lang"] == "unknown", "sentiment"] = df.loc[
    df["lang"] == "unknown", "sentiment"
].apply(lambda x: [])
df["sentiment_summary"] = df["sentiment"].apply(
    lambda x: sum(x) if isinstance(x, list) else 0
)


df.set_index("index", inplace=True)
df.to_csv("youtube_data_translated.csv")

game_name_db = game_name.replace(" ", "_")
# zapisywanie do bazy
try:

    conn = pg.connect(
        host="localhost",
        database="yt_data",
        user="postgres",
        password="123",
        port="5432",
    )

    create_table_query = f"""CREATE TABLE IF NOT EXISTS {game_name_db}_data (
        index TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        publishedAt DATE NOT NULL,
        channelName TEXT NOT NULL,
        duration FLOAT NOT NULL,
        viewCount BIGINT NOT NULL,
        likeCount BIGINT NOT NULL,
        commentCount BIGINT NOT NULL,
        comments TEXT[],
        lang TEXT[],
        translated_comments TEXT[],
        sentiment INTEGER[],
        sentiment_summary INTEGER,
        refresh_date DATE NOT NULL DEFAULT CURRENT_DATE
    );"""

    cursor = conn.cursor()
    cursor.execute(create_table_query)
    conn.commit()

    for idx, row in df.iterrows():
        if check_if_video_exists(cursor, idx):
            update_video_data(
                cursor,
                idx,
                row["viewCount"],
                row["likeCount"],
                row["commentCount"],
                row["comments"],
                row["lang"],
                row["translated_comments"],
                row["sentiment"],
                row["sentiment_summary"],
            )
        else:
            insert_video_data(
                cursor,
                idx,
                row["title"],
                row["publishedAt"],
                row["channelName"],
                row["duration"],
                row["viewCount"],
                row["likeCount"],
                row["commentCount"],
                row["comments"],
                row["lang"],
                row["translated_comments"],
                row["sentiment"],
                row["sentiment_summary"],
            )
    conn.commit()
    cursor.close()
    conn.close()

except Exception as e:
    print(f"Cannot connect: {e}")

a_input = input("Do you want to run analysis? (y/n): ").lower()
if a_input == "y":
    analysis.database_analysis(game_name_db)
end_time = time.time()

print(df)
print(f"Runtime: {end_time - start_time} seconds")
