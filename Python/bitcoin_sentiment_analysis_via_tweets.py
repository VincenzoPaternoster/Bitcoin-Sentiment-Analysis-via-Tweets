# -*- coding: utf-8 -*-
"""

# Bitcoin Sentiment Analysis via Tweets

## Library
"""

!pip install -U textblob
!pip install langdetect
!pip install emoji

## Library

### Pyspark
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import col,udf,to_date,avg,lag,log
from pyspark.sql.types import StringType

### Sentiment analysis and NLP
from textblob import TextBlob
from langdetect import detect
import emoji
import re

### Basic lib
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt

### Sklearn
from sklearn.preprocessing import MinMaxScaler

"""## 0. Upload dataset"""

## Upload dataset
# Download the file into /tmp/
!wget -O /tmp/bitcoin_tweets.csv https://proai-datasets.s3.eu-west-3.amazonaws.com/bitcoin_tweets.csv

import pandas as pd
dataset = pd.read_csv('/tmp/bitcoin_tweets.csv', delimiter=",")

spark = SparkSession.builder.getOrCreate()
spark_df = spark.createDataFrame(dataset)

## Check type of columns
spark_df.printSchema() # When I'll create the line chart, I'll convert the ‘timestamp column’ from string to date.

"""## 1. First requests

1. Classify daily tweets about Bitcoin as positive, negative, or neutral.

2. Plot a time series showing daily counts of each sentiment to visualize public opinion trends.

#### 1.1 Classify daily tweets

##### 1.1.1 Data preprocessing
"""

##  Check if there are any NULL values in "text" column
display(spark_df.filter(col("text").isNull()))

## Detect languages

## Detect language and return "unknown" if the language is not supported
def detect_lang(text):
    try:
        return detect(text)
    except:
        return "unknown"

## Create UDF (User Defined Function) to use new python function with spark dataframe
detect_lang_udf = udf(detect_lang, StringType())

## Create a new column with the language detected
df_lang=spark_df.withColumn("lang", detect_lang_udf(col("text")))

## See dataset with new column "lang"
display(df_lang)

## How many tweets are in each language?
df_lang.groupBy("lang").count().sort("count",ascending=False).show()

## I'll use only english tweets (75.135) because I am going to use TextBlob to analyze sentiment and it only supports English language
df_eng=df_lang.filter(col("lang")=="en")

## Delete retweets because I only want original tweets
df_ret=df_eng.filter(col("text").startswith("RT")==False)

## Check if there are duplicate user and id tweet
df_ret.groupBy(["user","id"]).count().sort("count", ascending=False).show()

## Delete duplicate
df_fin=df_ret.dropDuplicates(["id"])

## Remove emoji and url from "text" column

# NOTE: I used this function from : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b

def remove_emoji_url(string):
    # Remove emoji
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        u"\U00002500-\U00002BEF"  # chinese chars
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"
        u"\u3030"
        u"\u205A"
        "]+", flags=re.UNICODE)

    # Remove url
    url_pattern = re.compile(r"https?://\S+|www\.\S+") # remove url such as http or https and www.

    # Remove special characters
    spec_char_pat=re.compile(r"[^\w\s.,?!]")

    # Apply all the functions
    string = emoji_pattern.sub(r"", string)   # Removing emoji
    string = url_pattern.sub(r"", string)     # Removing url
    string = spec_char_pat.sub(r"", string)   # Removimg special characters
    return string

## From python function to spark function with UDF (User Defined Function)
remove_emoji_url_udf=udf(remove_emoji_url,StringType())

# Remove emoji, url and special characters
df_clean=df_fin.withColumn("text", remove_emoji_url_udf(col("text")))

"""##### 1.1.2 Sentiment Analysis"""

#Sentiment analysis

# Create function to define sentiment polarity of tweet
def get_sentiment(df_clean):

	analysis = TextBlob(df_clean)

    ## 1:positive
	if analysis.sentiment.polarity > 0:
		return 1

    ## 0:negative
	elif analysis.sentiment.polarity < 0:
		return 0

    ## 2:neutral
	else: return 2

# Create and convert from Python function to PySpark function
sentiment_udf = udf(get_sentiment, StringType())

## Create the column "sentiment_label" in the new dataset df_fin
df_fin =df_clean.withColumn("sentiment_label", sentiment_udf(col("text")))

### How many tweets are positive (1), negative (0) and neutral (2)?
df_fin.groupBy("sentiment_label").count().show()

"""#### 1.2 Graph showing sentiment of tweets per daily tweet"""

## Create a graph with tweet sentiment grouped by daily tweet

# Get the date
df_fin=df_fin.withColumn("date",to_date(col("timestamp")))

## Count labels for each day
df_count_lab = df_fin.groupBy("date", "sentiment_label").count().orderBy("date")

## Convert to pandas dataframe
df_count_pd=df_count_lab.toPandas()

## Create a pivot table with columns for each label in order to obtain the count of tweets with sentiment labels for each date
df_count_pd['date'] = pd.to_datetime(df_count_pd['date'])

## Pivot table
pivot_lab = df_count_pd.pivot(index='date', columns='sentiment_label', values='count').fillna(0)

## Reset index (date) because "date" column results as object type after pivot operation
pivot_lab = pivot_lab.reset_index()

## Convert column "date" to date type
pivot_lab['date'] = pivot_lab['date'].dt.date

## Show pivot_lab
display(pivot_lab)

## LINE CHART

## Get max number of tweet in order to write on line chart
max0_count=pivot_lab["0"].max()
max1_count=pivot_lab["1"].max()
max2_count=pivot_lab["2"].max()
display([max0_count,max1_count,max2_count])
# Obtain date of maximum number of tweet for each type
max0_date=pivot_lab["date"][pivot_lab["0"] == pivot_lab["0"].max()]
max1_date=pivot_lab["date"][pivot_lab["1"] == pivot_lab["1"].max()]
max2_date=pivot_lab["date"][pivot_lab["2"] == pivot_lab["2"].max()]
display([max0_date,max1_date,max2_date])

## Create linechart to show trend of sentiment
pivot_lab.plot.line(
    x="date",
    y=["0", "1", "2"],
    figsize=(12, 6),
    title="Sentiment analysis grouped by date",
    logy=True, # log scale to reduce the scale of the graph
    alpha=0.7,
    color=['red','green','lightblue']
)

## Name axis and legend
plt.legend(["Negative Tweet","Positive Tweet", "Neutral Tweet"],loc="upper left")
plt.xlabel("Date")
plt.ylabel("Tweet Count (log scale)")

## Take note of the maximum number of tweets for each type of sentiment
plt.annotate(f"Max negative \n {max0_count}", xy=(max0_date,max0_count))
plt.annotate(f"Max positive \n {max1_count}", xy=(max1_date,max1_count),xytext=(max1_date,max1_count-1000),ha="right")
plt.annotate(f"Max neutral \n {max2_count}", xy=(max2_date,max2_count))

## Axis limits
plt.ylim(1,100000)
plt.xlim(pd.to_datetime("2015-01-01"), pd.to_datetime("2019-08-31"))

## General setting of chart
plt.grid(True)
plt.tight_layout()
plt.show()

### I chose to show only the period from 2015 to 2019 because there are few tweets from 2009 to 2014

#### NOTE: UserWarning derives from my error to define limit of y as 0,100000.

"""## 2. Second request

-  Are there any differences in the number of like and replies between positive and negative tweet?

"""

## Create the two subset
pos=df_fin.filter(df_fin.sentiment_label==1)
neg=df_fin.filter(df_fin.sentiment_label==0)

## Use toPandas in order to get descriptive statistics and create graphs
pos_pd=pos.toPandas()
neg_pd=neg.toPandas()

# Descriptive statistics
print(pos_pd.describe())
print(neg_pd.describe())

# Custom quantiles

print(pos_pd[["likes","replies","retweets"]].quantile([0.25,0.5,0.75,0.90,0.95,0.99]))
print(neg_pd[["likes","replies","retweets"]].quantile([0.25,0.5,0.75,0.90,0.95,0.99]))

"""
##### **Graphs**
"""
## Show the two distributions

# Create a dataset that combines the two sample in order to show
# if there are difference between positive and negative tweets per number of like and replies
df_combined = pd.concat([pos_pd, neg_pd], ignore_index=True)

# Create a dataset without zero likes in order to use KDE and log scale
## Engagement data (such as likes, replies, etc.) usually has many small values, which is why it is useful to
## apply a logarithmic scale to better visualise all areas of a distribution
df_nozero_like=df_combined[df_combined["likes"]>0]
df_nozero_repl=df_combined[df_combined["replies"]>0]

# KDE plots are used for exploratory analysis of distribution shape
## Create "like" KDE graph
sbn.kdeplot(data=df_nozero_like, x="likes", hue="sentiment_label", log_scale=True, bw_adjust=0.5)
plt.legend(["Negative Tweet","Positive Tweet"])
plt.xscale("symlog",linthresh=1) ## symlog manage the scale of the graph
plt.title("The number of likes for positive and negative tweet")
plt.show()

## Create "replies" KDE graph
sbn.kdeplot(data=df_nozero_repl,x="replies",hue="sentiment_label",log_scale=True,bw_adjust=0.5)
plt.legend(["Negative Tweet","Positive Tweet"])
plt.xscale("symlog",linthresh=1) ## symlog manage the scale of the graph
plt.title("The number of replies for positive and negative tweet")
plt.show()

# Box plots are used to compare dispersion and highlight outliers.

## Boxplot like
plt.boxplot([pos_pd["likes"],neg_pd["likes"]],tick_labels=["Positive","Negative"])
plt.title("Boxplot of likes between positive and negative tweets")
plt.xlabel("Sentiment label")
plt.ylabel("Number of likes")
plt.yscale("log")
plt.ylim(1,100000)
plt.grid(True)
plt.show()

## Boxplot replies
plt.boxplot([pos_pd["replies"],neg_pd["replies"]],tick_labels=["Positive","Negative"])
plt.title("Boxplot of replies between positive and negative tweets")
plt.xlabel("Sentiment label")
plt.ylabel("Number of replies")
plt.yscale("log")
plt.ylim(1,100000)
plt.grid(True)
plt.show()

"""
## 3. Third request

- Verify whether the fluctuation in the value of Bitcoin affects the change in sentiment in tweets about Bitcoin.

## Upload bitcoin dataset
## This dataset shows the value of bitcoin by date.
## For each date, the opening, closing, maximum and minimum values and volume are shown.
"""

### Upload dataset using pandas dataframe
df_bit=pd.read_csv(r"https://raw.githubusercontent.com/Profession-AI/progetti-big-data/refs/heads/main/Analisi%20del%20consenso%20sul%20Bitcoin/BTC-USD.csv")

### Convert pandas dataframe in pyspark dataframe
df_bit_sp=spark.createDataFrame(df_bit)
df_bit_sp.createOrReplaceTempView("df_bit_sp")

## Create date column without time
df_bit_sp=df_bit_sp.withColumn("only_date",to_date(col("Date")))

# Drop "Date" column
df_bit_sp=df_bit_sp.drop("Date")

# Filter by date both dataset in order to get the same period of time

## Filter the Bitcoin dataset by date in order to compare these values with the date of the sentiment dataset.
df_bit_fin=df_bit_sp.filter((df_bit_sp["only_date"]>="2014-09-17") & (df_bit_sp["only_date"]<="2019-05-27")).sort("only_date",ascending=True)

## Filter with the same date limit for the sentiment dataset
df_fin_filt=df_fin.filter((df_fin["date"]>="2014-09-17") & (df_fin["date"]<="2019-05-27")).sort("date",ascending=True)

# Create a difference column between OPEN and CLOSE to obtain the change in the price of Bitcoin

## Calculate bitcoin price variation
df_bit_fin=df_bit_fin.withColumn("var_price",log(col("Close"))-log(col("Open")))

# Calculate sentiment variation so as to compare it with Bitcoin price variation

## 1. Daily sentiment average
df_sent1=df_fin_filt.groupBy("date").agg(avg("sentiment_label").alias("sent_avg"))

## 2. Get column with all the previous sentiment
window=Window.orderBy("date")
df_sent2=df_sent1.withColumn("prec_value",lag("sent_avg").over(window))

## 3. Calculate variation between daily sentiment average and the previous sentiment
df_sent_fin=df_sent2.withColumn("var_sent",col("sent_avg")-col("prec_value"))

# Joim two data sets so that they can be represented in a chart
join_df=df_sent_fin.join(df_bit_fin,df_sent_fin["date"]==df_bit_fin["only_date"],"inner")

## After window function there is a null value (the first of the column)
## I decided to remove the null value because in this context, having a previous missing value does not make sense
join_df=join_df.filter(col("only_date")!="2014-10-03")

### Convert pypspark dataframe in pandas dataframe
pddf=join_df.toPandas()

# Calculate the correlation between sentiment variation and bitcoin price fluctuation
## Is there an association between sentiment and bitcoin price variations?

### Create correlation matrix (Heatmap)
plt.figure(figsize=(8,6))
sbn.heatmap(pddf[["var_sent",'var_price']].corr(method="kendall"),annot=True,cmap="Greens") #I used Kendall's tau because linearity assumption is not checked
plt.title("Correlation between variation of sentiment and bitcoin price")
plt.show()

"""##### **Note on correlatin matrix output**

###### The correlation result (t: -0.042) shows that there is no correlation between the two variables.
"""

## Normalise two columns (for RegPlot) so that they have the same variable scale and display the correct result
scaler = MinMaxScaler()
pddf[["var_sent_norm", "var_price_norm"]] = scaler.fit_transform(pddf[["var_sent", "var_price"]])

## Regplot (scatterplot with line)
sbn.regplot(x=pddf["var_sent_norm"],y=pddf["var_price_norm"],ci=None, line_kws={"color": "red"})
plt.xlabel("Sentiment variation")
plt.ylabel("Bitcoin variation")
plt.title("Scatterplot: Bitcoin variation and Sentiment variation")
plt.show()

"""##### **Note on RegPlot output**

##### The relationship is slightly negative. The result confirms the output of the correlation matrix. Therefore, based on these analyses, there are no associations between sentiment and changes in the price of Bitcoin.
"""

## Create a chart showing both sentiment fluctuations and the value of Bitcoin by date
## Line chart
pddf.plot.line(
    x="date",
    y=["var_sent_norm", "var_price_norm"],
    label=["Sentiment Variation","Bitcoin Value Variation"],
    figsize=(14, 6),
    title="Normalized Sentiment and Bitcoin variation",
    color=["red", "green"],
    alpha=0.7
)
plt.legend()
plt.show()

