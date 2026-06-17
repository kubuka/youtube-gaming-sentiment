import psycopg2 as pg
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker


def get_data_from_db(game_name_db):

    try:
        conn = pg.connect(
            host="localhost",
            database="yt_data",
            user="postgres",
            password="",
            port="5432",
        )
        cursor = conn.cursor()
        query = f"SELECT * FROM {game_name_db}_data"
        df = pd.read_sql_query(query, conn)
        cursor.close()
        conn.close()
        return df
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None


def database_analysis(game_name_db):
    df = get_data_from_db(game_name_db)
    game_name = game_name_db.replace("_", " ")
    if df is not None and not df.empty:
        df["publishedat"] = pd.to_datetime(df["publishedat"]).dt.strftime("%d.%m.%y")
        df["refresh_date"] = pd.to_datetime(df["refresh_date"])

        df["duration_category"] = pd.cut(
            df["duration"],
            bins=[-1, 300, 1200, float("inf")],
            labels=["short", "medium", "long"],
        )

        def k_format(x, pos):
            if x >= 1e9:
                return f"{x*1e-9:.1f}B"
            elif x >= 1e6:
                return f"{x*1e-6:.1f}M"
            elif x >= 1e3:
                return f"{x*1e-3:.1f}k"
            return f"{x:.0f}"

        sns.set_palette("pastel")

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

        # popularność w czasie
        df_trend = df.groupby("publishedat")[["viewcount", "likecount"]].sum()
        axes[0, 0].set_title(f"View and Like Count Trends over Time for {game_name}")
        axes[0, 0].set_xlabel("Dates")
        axes[0, 0].set_ylabel("View Count")
        axes[0, 0].grid(True)
        axes[0, 0].xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
        axes[0, 0].yaxis.set_major_formatter(ticker.FuncFormatter(k_format))
        axes[0, 0].yaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
        axes[0, 0].tick_params(axis="x", rotation=45)

        axes[0, 0].plot(
            df_trend.index, df_trend["viewcount"], label="Views", linestyle="-"
        )
        axes[0, 0].plot(
            df_trend.index, df_trend["likecount"], label="Likes", linestyle="--"
        )
        axes[0, 0].legend()

        # rozklad sentymentu
        df_exploded_sentiment = df.explode("sentiment")
        sentiment_counts = df_exploded_sentiment["sentiment"].value_counts()
        sentiment_counts.index = sentiment_counts.index.map(
            {1: "Positive", -1: "Negative", 0: "Neutral"}
        )
        print(sentiment_counts)
        axes[0, 1].pie(
            sentiment_counts,
            labels=sentiment_counts.index,
            autopct="%1.1f%%",
            startangle=90,
        )
        axes[0, 1].set_title(f"Sentiment disribution for {game_name}")

        # like to view correlation
        axes[1, 0].set_title(f"Like to View Correlation for {game_name}")
        axes[1, 0].set_xlabel("View Count")
        axes[1, 0].set_ylabel("Like Count")
        axes[1, 0].xaxis.set_major_formatter(ticker.FuncFormatter(k_format))
        axes[1, 0].yaxis.set_major_formatter(ticker.FuncFormatter(k_format))
        axes[1, 0].xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
        axes[1, 0].yaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
        axes[1, 0].tick_params(axis="x", rotation=45)
        axes[1, 0].grid(True)
        axes[1, 0].scatter(df["viewcount"], df["likecount"], alpha=0.8)
        correlation = df["viewcount"].corr(df["likecount"])
        axes[1, 0].text(
            0.05,
            0.95,
            f"Correlation: {correlation:.2f}",
            transform=axes[1, 0].transAxes,
            fontsize=12,
            verticalalignment="top",
        )

        df_duration_mean = df.groupby("duration_category")[["viewcount"]].count()
        df_duration_mean["viewcount"] = (
            df_duration_mean["viewcount"] / df_duration_mean["viewcount"].sum() * 100
        )
        axes[1, 1].set_title(f"Video Duration Distribution for {game_name}")
        axes[1, 1].set_xlabel("Duration Category")
        axes[1, 1].set_ylabel("Percentage of Videos (%)")
        axes[1, 1].yaxis.set_major_formatter(ticker.PercentFormatter())
        axes[1, 1].bar(
            df_duration_mean.index,
            df_duration_mean["viewcount"],
            color=sns.color_palette("pastel"),
        )
        print(df_duration_mean)

        plt.tight_layout()
        plt.show()
