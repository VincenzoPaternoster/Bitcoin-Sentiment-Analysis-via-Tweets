# 📌 Bitcoin Sentiment Analysis via Tweets

The aim of the project is to perform sentiment analysis on the collected data and understand how opinions about Bitcoin vary over time. 
In addition, user engagement on social media and any correlation with the price of Bitcoin will be evaluated.

---

## 📂 Repository Structure
```
project/
│── data/
│── images/
│── python/ ## contains notebook
|__ README.md
```
---

## 🎯 Project objectives
- 1 Doing sentiment analysis on tweets (negative,positive, neutral)
- 2 Show daily trend of sentiment tweets through a linechart
- 3 Check if:
  
    a. Negative tweets tend to obtain more like compared to positive tweets
  
    b. Negative tweets receive more answers compared to positive tweets
  
- 4 Test association between bitcoin value variation and sentiment tweet variation

---

## 🗂️ Dataset
**Source:** Profession AI 
**Period examined:** 2014-2019  
**Dimension of dataset:** 75.135 (english tweets)

---

### 🌗 Main variables

#### Dataset with tweets
| Variable | Description |
|----------|-------------|
|id | User identifier  |
|user | Username  |
|replies  | Number of replies per tweet  |
|likes  | Number of likes per tweet  |
|retweets  | Number of retweets |
|text  |Content of tweet |
|timestamp  | Date and time of tweet |

#### Dataset with Bitcoin values
| Variable | Description |
|----------|-------------|
|Date | Bitcoin values date |
|Open | Bitcoin opening value|
|High | Maximum value of Bitcoin |
|Low  | Minimum value of Bitcoin |
|Close| Bitcoin close value |
|Volume  |Total amount of Bitcoing sold and bought |

---

## 🧹 Data Cleaning
Main operations:
- Check for missing values
- Detect language
- Delete duplicates and retweets
- Filter tweet by English language
- Remove emojy, url, special characters from the text

---

## 📊 Methodology
Description:
- Used techniques (Sentiment Analysis, Normalizzation, Window function)
- Main used libraries (PySpark,Pandas,Seaborn,Sklearn)
---

## 🔍 Key results
- 1  The dataset with tweets shows a higher number of tweets with positive (32,005) and neutral (33,763) sentiment than those with negative sentiment (9,010).
- 2  Sentiment variations peaked for all types of sentiment on 10 May 2019 [LineChart](https://github.com/VincenzoPaternoster/Bitcoin-Sentiment-Analysis-via-Tweets/blob/main/Image/Line_chart_countsentiment.png)
- 3  The distribution of likes and replies is heavily skewed to the right, dominated by values equal to zero, and consequently the median is zero in both sentiment classes, indicating equality between tweet types. [Boxplot](https://github.com/VincenzoPaternoster/Bitcoin-Sentiment-Analysis-via-Tweets/blob/main/Image/BoxPlot.png) and [KDEPlot](https://github.com/VincenzoPaternoster/Bitcoin-Sentiment-Analysis-via-Tweets/blob/main/Image/KDEPlot.png)
- 4  The variation in sentiment of tweets about Bitcoin and the variation in the daily price of Bitcoin in this dataset are not associated. [Correlation matrix](https://github.com/VincenzoPaternoster/Bitcoin-Sentiment-Analysis-via-Tweets/blob/main/Image/CorrPlot.png) and [RegPlot](https://github.com/VincenzoPaternoster/Bitcoin-Sentiment-Analysis-via-Tweets/blob/main/Image/RegPlot.png)

---

## 🧠 Conclusions
- This project allowed me to understand how to use the basic commands of the PySpark library, and I learned how to manage big data with it. In addition, I learned how to use natural language processing techniques (such as sentiment analysis) and SQL functions (e.g., Window functions) through PySpark with large amounts of data.


---

## 🛠️ Tools
- Python (pandas, PySpark,seaborn,sklearn)
- Jupyter / Google Colab

---

## 📬 Contacts

- **Vincenzo Paternoster**
- Email: vincenzopaternoster99@gmail.com
- LinkedIn: www.linkedin.com/in/vincenzo-paternoster
