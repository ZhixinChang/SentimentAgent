import jieba.posseg as pseg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import os

def get_stop_words():

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stop_words/stop_words.txt"), 'r', encoding='utf-8') as file:
        # 读取全部内容
        stop_words_set = set(file.read().split('\n'))
    return stop_words_set


def chinese_preprocess(text):
    stop_words_set = get_stop_words()
    words = pseg.lcut(text)

    return " ".join(
        [
            w.word
            for w in words
            if w.flag not in ["x", "m", "q"]
               and w.word not in stop_words_set
               and len(w.word) > 1
        ]
    )


def get_top_keywords(kmeans, tfidf, n_clusters, n_keywords=10):
    centroid = kmeans.cluster_centers_[n_clusters]
    feature_names = tfidf.get_feature_names_out()
    top_indices = centroid.argsort()[-n_keywords:][::-1]
    return [feature_names[i] for i in top_indices]


def get_pre_classified_summary(kmeans, tfidf, pre_classified_df, n_clusters, input_col, score_col):
    pre_classified_summary = pd.DataFrame(
        {
            "cluster": np.arange(n_clusters),
            "keywords": [
                ", ".join(get_top_keywords(kmeans, tfidf, n_clusters=i)) for i in range(n_clusters)
            ],
            "sample_cases": [
                "\n".join(
                    [
                        f"{sample_case}"
                        for sample_case in pre_classified_df[pre_classified_df["cluster"] == i]
                                           .sort_values(by="score", ascending=True)[input_col]
                                           .to_list()[:5]
                    ]
                )
                for i in range(n_clusters)
            ],
            "comment_cnt": [
                pre_classified_df[pre_classified_df["cluster"] == i].shape[0]
                for i in range(n_clusters)
            ],
            "comment_pct": [
                pre_classified_df[pre_classified_df["cluster"] == i].shape[0] / len(pre_classified_df)
                for i in range(n_clusters)
            ],
        }
    )
    if score_col is not None:
        pre_classified_summary[score_col] = [
                pre_classified_df[pre_classified_df["cluster"] == i]["score"].astype('float64').mean()
                for i in range(n_clusters)
            ]
    pre_classified_summary[['体验问题', '推理原因']] = ''

    return pre_classified_summary


def pre_classified_fit(df, input_col, pre_cluster_num=20, score_col=None):
    pre_classified_df = df.copy()
    pre_classified_df["corpus"] = pre_classified_df[input_col].map(chinese_preprocess)

    # TF-IDF向量化
    tfidf = TfidfVectorizer(max_features=1000)
    X = tfidf.fit_transform(pre_classified_df["corpus"])
    # 计算不同簇数下的SSE
    sse = []
    cluster_range = range(1, pre_cluster_num + 1)
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=1)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)  # inertia_表示SSE

    # 绘制肘部法则图
    plt.plot(cluster_range, sse, marker='o')
    plt.xlabel("n_clusters (k)")
    plt.ylabel("SSE")
    plt.title("The elbow rule determines the optimal number of clusters")
    plt.xticks(cluster_range)
    plt.grid()
    plt.show()

    # 4. KMeans聚类
    n_clusters = int(input('请结合肘部法则输入预分类数量：'))
    kmeans = KMeans(n_clusters=n_clusters, random_state=1)
    pre_classified_df['cluster'] = kmeans.fit_predict(X)

    pre_classified_summary = get_pre_classified_summary(kmeans, tfidf, pre_classified_df, n_clusters, input_col, score_col)

    return pre_classified_df, pre_classified_summary


def get_classified_summary(df, input_col):
    classified_df = df.copy()
    classified_summary = classified_df.groupby(by=input_col)[[input_col]].agg({input_col: 'count'}).rename(
        columns={input_col: 'comment_cnt'}).reset_index()
    classified_summary['comment_pct'] = classified_summary['comment_cnt'] / len(classified_df)
    classified_summary['推理原因'] = ''
    return classified_df, classified_summary
