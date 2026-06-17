# YouTube-Gaming-Sentiment-Analysis 🎮🔍
This project is a powerful tool for analyzing video data from YouTube, focused on content related to video games. The script automatically fetches videos, comments, and statistics from YouTube, then uses advanced NLP  models to perform sentiment analysis on those comments. All the data is stored in a PostgreSQL database, allowing for easy data refreshes and further analysis. It's a comprehensive system that lets you measure how audiences are reacting to a particular video game and track the evolution of public opinion over time.

# 📝 Scripts and Functionality Overview
1. [Fetching Data from YouTube](#Fetching-Data-from-YouTube)
2. [Data Processing and Sentiment Analysis](#Data-Processing-and-Sentiment-Analysis)
3. [Saving Data to PostgreSQL](#Saving-Data-to-PostgreSQL)
4. [Example Analysis and Insights](#Example-Analysis-and-Insights)
5. [Summary](#Summary)

# Fetching Data from YouTube
The heart of the project is the `fetch_yt_data()` function, which communicates with the YouTube Data API v3. Its purpose is to collect a comprehensive set of information about videos related to a given game.

## Key Steps
- ***Search***: The script searches for videos on YouTube based on the game's name, sorting the results by date to get the most recent content.

- ***Filtering***: It fetches videos from the selected date up to current day.

- ***Categories***: The search is limited to two video duration categories: medium (4-20 minutes) and long (over 20 minutes), which allows for a varied analysis of content.

- ***Statistics***: For each video, it collects statistics like view count (viewCount), like count (likeCount), and comment count (commentCount), as well as the duration.

- ***Comments***: The most important element is the retrieval of up to 200 comments per video, which forms the basis for the sentiment analysis.

## Code
```python
def fetch_yt_data(game_name,video_duration):
    """
    Fetches YouTube video data for a given game name.
    Returns a DataFrame with video details.
    """
    youtube = build(
        'youtube', 
        'v3', 
        developerKey='AIza'

    filter_date = date.today() - timedelta(days=int(time_frame)) 
    next_page_token = None
    old_video = True
    df = pd.DataFrame()
    df['comments'] = ''

    while old_video:
        requests = youtube.search().list(
            part='snippet',
            q=game_name,
            type='video',
            order='date',
            videoDuration=video_duration,
            eventType='completed',
            pageToken=next_page_token,
            maxResults=50
        )
        vid_ids = []
        response = requests.execute()
        for item in response['items']:
            if item['snippet']['publishedAt'] >= filter_date.isoformat():
                video_id = item['id']['videoId']
                vid_ids.append(video_id)
                df.loc[video_id, 'title'] = item['snippet']['title']
                df.loc[video_id, 'publishedAt'] = item['snippet']['publishedAt'].split('T')[0]
                df.loc[video_id, 'channelName'] = item['snippet']['channelTitle']       
            else:
                old_video = False
                break

        requests = youtube.videos().list(
            part='statistics,contentDetails',
            id=','.join(vid_ids)
        )

        vidresponse = requests.execute()
        for item in vidresponse['items']:
            video_id = item['id']
            if video_id in df.index:
                df.loc[video_id, 'duration'] = parse_duration(item['contentDetails']['duration']).total_seconds()
                df.loc[video_id, 'viewCount'] = int(item['statistics'].get('viewCount', 0))
                df.loc[video_id, 'likeCount'] = int(item['statistics'].get('likeCount', 0))
                df.loc[video_id, 'commentCount'] = int(item['statistics'].get('commentCount', 0))

            else:
                print(f"No stats for ID: {video_id}")
        
  
        for video_id in vid_ids:
            comments_next_page_token = None
            comments_list = []
            try:
                while True:
                    comment_response = youtube.commentThreads().list(
                        part='snippet',
                        videoId=video_id,
                        textFormat='plainText',
                        pageToken=comments_next_page_token,
                        maxResults=100
                    )
                    
                    
                    requests = comment_response.execute()
                    for item in requests['items']:
                        comment_snippet = item['snippet']['topLevelComment']['snippet']
                        comments_list.append(comment_snippet['textDisplay'])
                    
                    comments_next_page_token= requests.get('nextPageToken', None)

                    if len(comments_list) >= 200:
                        df.at[video_id, 'comments'] = comments_list  
                        break

                    if not comments_next_page_token:
                        df.at[video_id, 'comments'] = comments_list         
                        break

            except HttpError as error:
                if error.resp.status == 403:
                    print(f"Error 403: {video_id}")
                    df.at[video_id, 'comments'] = [] 
                    

        next_page_token = response.get('nextPageToken')
        time.sleep(1)

        if not next_page_token:
            break

    next_page_token = None
    return df
```

# Data Processing and Sentiment Analysis
After fetching the raw data, it's crucial to prepare it for analysis. This step is fully automated and includes:

- ***Comment Normalization***: Removing unwanted characters, mentions, hashtags, and emojis, ensuring clean text for analysis.
```python
def normalize_comment(comment):

    comment = comment.lower()
    comment = comment.replace('\n', ' ').strip()
    comment = re.sub(r'http\S+', '', comment)
    comment = re.sub(r'@\w+', '', comment)  
    comment = re.sub(r'#[\w-]+', '', comment) 

    comment = emoji.demojize(comment) 
    comment = re.sub(r':\w+:', '', comment)
    comment = re.sub(r'[^\w\s.,?!-]', '', comment, flags=re.UNICODE)

    timestamp_pattern = r'\b(\d{1,2}:\d{2}(:\d{2})?)\b'
    comment = re.sub(timestamp_pattern, '', comment)
    return comment
```

- ***Language Detection***: Each comment is checked for its language.
```python
def safe_detect(text):
    try:
        if text and len(text.strip()) > 3:
            return detect(text)
        else:
            return 'unknown'
    except LangDetectException:
        return 'unknown'
```

- ***Translation***: Using a Hugging Face Transformers pipeline and the Helsinki-NLP/opus-mt-mul-en model, all non-English comments are translated, allowing for a unified analysis.
```python
batch_size = 50
translated_texts = []
for i in range(0, len(to_translate), batch_size):
    batch = to_translate[i:i + batch_size] 
    batch_results = translation_pipeline(batch)
    translated_texts.extend([res['translation_text'] for res in batch_results]) 
    print(f"translated: {i + len(batch)} / {len(to_translate)}")
```
- ***Sentiment Analysis***: The translated comments are analyzed using the sentiment-analysis model from Hugging Face, which classifies them as positive (POSITIVE), negative (NEGATIVE), or neutral. The sentiment is then quantified (1 for positive, -1 for negative, 0 for neutral).
```python
for i in range(0, len(comments_to_analyze), batch_size_sentiment):
    batch_sentiment = comments_to_analyze[i:i + batch_size_sentiment]
    batch_results_sentiment = sentiment_pipeline(batch_sentiment, max_length=512, truncation=True)
    sentiment_results_list.extend(batch_results_sentiment)
    print(f"sentiment: {i + len(batch_sentiment)} / {len(comments_to_analyze)}")

sentiments = [1 if result['label'] == 'POSITIVE' 
                else -1 if result['label'] == 'NEGATIVE'
                else 0  for result in sentiment_results_list]
```

- ***Sentiment Summary***: For each video, an overall sentiment score is calculated, allowing for a quick determination of whether the general reception is positive or negative.
```python
df['sentiment_summary'] = df['sentiment'].apply(lambda x: sum(x) if isinstance(x, list) else 0)
```

## Disclaimer
The translation and sentiment analysis models used in this project are free, open-source models designed to run on a local machine. While they are highly effective, they may not be as accurate as paid ones. For more precise results, it's recommended to use these more advanced, cloud-based solutions.

# Saving Data to PostgreSQL
The processed data is then saved to a local PostgreSQL database.

- ***Table Creation***: The script automatically creates a table named [game_name]_data if it doesn't already exist.
  ```python
  create_table_query = (f"""CREATE TABLE IF NOT EXISTS {game_name_db}_data (
        index TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        publishedAt DATE NOT NULL,
        channelName TEXT NOT NULL,
        duration FLOAT NOT NULL,
        (...)
  ```
- ***Update/Insert Logic***: The check_if_video_exists() function checks if a video with a given ID (index) is already in the database.
If it exists, it updates the statistics and comments, which is perfect for data refreshes.
If it does not exist, it inserts a new record.
```python
def check_if_video_exists(cursor,index):
    query = (f""" SELECT index FROM {game_name_db}_data WHERE index = %s""")
    cursor.execute(query, (index,))
    return cursor.fetchone() is not None

def update_video_data(cursor, index,viewCount,likeCount,commentCount, comments, lang, translated_comments, sentiment, sentiment_summary):
    query = (f""" UPDATE {game_name_db}_data 
                  SET viewCount = %s, likeCount = %s, commentCount = %s, comments = %s, lang = %s, translated_comments = %s, sentiment = %s, sentiment_summary = %s, refresh_date = CURRENT_DATE
                  WHERE index = %s""")
    cursor.execute(query, (viewCount,likeCount,commentCount,comments, lang, translated_comments, sentiment, sentiment_summary, index))

def insert_video_data(cursor, index, title, publishedAt, channelName, duration, viewCount, likeCount, commentCount, comments, lang, translated_comments, sentiment, sentiment_summary):
    query = (f""" INSERT INTO {game_name_db}_data (index, title, publishedAt, channelName, duration, viewCount, likeCount, commentCount, comments, lang, translated_comments, sentiment, sentiment_summary, refresh_date)
                  VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_DATE)""")
    cursor.execute(query, (index, title, publishedAt, channelName, duration, viewCount, likeCount, commentCount, comments, lang, translated_comments, sentiment, sentiment_summary))

```

# Example Analysis and Insights
Once the data is saved, your database transforms into a powerful information hub, which you can explore using the `analysis.py` script. This script automatically connects to the database, retrieves the processed data, and generates four key visualizations, enabling in-depth analysis of trends and public opinion.

## Functionality
- ***View and Like Count Trend over Time***: This shows how video popularity changes over the defined time frame.
- ***Sentiment Distribution***: A pie chart showing the breakdown of comments into positive, negative, and neutral categories.
- ***Like-to-View Correlation***: This scatter plot analyzes whether videos with a high number of views also have a high number of likes.
- ***Video Duration Distribution***: This bar chart shows the percentage of videos in the dataset that fall into *short* (<5 minutes), *medium* (5-20 minutes), and *long* (>20 minutes) categories

<img width="1996" height="1592" alt="Zrzut ekranu 2025-09-15 o 19 55 03" src="https://github.com/user-attachments/assets/77694730-e0cf-41f3-8ead-60d2c13ec7fe" />

## 💡 Insights from the Visualizations
By looking at the visualizations, we can draw key conclusions:

- Popularity Dynamics: Prominent peaks on the trend graph can indicate new releases or significant events that generate high activity. (In this case, that specific spike in views was an announcement for the new expansion)

- Overall Community Mood: The dominance of a positive sentiment in the pie chart suggests that the game has a loyal and satisfied fan base.

- Content Quality: A strong correlation on the scatter plot between likes and views is a sign of high-quality videos that are not only watched but also positively rated.

- Preferred Formats: The bar chart showing video duration can reveal whether the community prefers short, "quick" clips or longer, more detailed reviews and gameplay sessions.

# Summary
This project is a robust tool for analyzing video game content on YouTube. It uses the YouTube Data API to fetch video statistics and comments, then employs NLP models from the Hugging Face library for language detection, translation, and sentiment analysis. The processed data is stored in a PostgreSQL database. To provide actionable insights, the system includes a dedicated `analysis.py` script that generates key visualizations, helping to understand community trends and public opinion with ease.

## What I Learned 
Developing this project provided a fantastic opportunity to sharpen my skills in several key areas:

- ***API Integration***: I gained hands-on experience in building complex, batched requests to efficiently handle large amounts of data from a public API.

- ***Practical NLP Implementation***: I learned how to use state-of-the-art NLP models for real-world tasks like cross-lingual sentiment analysis, which required handling data normalization, language detection, and translation pipelines.

- ***Database Management***: I developed a robust system for storing and managing data in a PostgreSQL database, including dynamic table creation and logic for updating existing records versus inserting new ones.

- ***Python Data Visualization***: I learned to transform raw, processed data into a compelling visual story using Matplotlib and Seaborn, creating charts that clearly communicate key insights and trends.

  

