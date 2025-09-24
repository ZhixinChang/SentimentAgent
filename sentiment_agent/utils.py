import jieba.posseg as pseg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import os


def get_simulated_data_by_llm():

    df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples/simulated_data_by_llm/domain_酒店_question_type_居住质量.txt"))

    return df


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


def get_pre_classified_summary(kmeans, tfidf, pre_classified_df, n_clusters, content_col, sentiment_col):
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
                        for sample_case in pre_classified_df[pre_classified_df["cluster"] == i][content_col].to_list()[:5]
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
    if sentiment_col is not None:
        pre_classified_summary[sentiment_col] = [
                pre_classified_df[pre_classified_df["cluster"] == i][sentiment_col].astype('float64').mean()
                for i in range(n_clusters)
            ]
    pre_classified_summary[['体验问题', '推理原因']] = ''

    return pre_classified_summary


def polynomial(x, a, b):
    return a * x + b


def detect_elbow_k_residual(k_values, inertia_values):
    coef, _ = curve_fit(polynomial, np.array(k_values)[[0, -1]], np.array(inertia_values)[[0, -1]])
    fitted_values = polynomial(k_values, *coef)
    residuals = np.abs(inertia_values - fitted_values)
    peak_index = np.argmax(residuals)
    return k_values[peak_index], coef


def pre_classified_fit(pre_classified_df, content_col, pre_cluster_num=20, sentiment_col=None):
    pre_classified_df = pre_classified_df.copy()
    pre_classified_df["corpus"] = pre_classified_df[content_col].map(chinese_preprocess)

    # TF-IDF向量化
    tfidf = TfidfVectorizer(max_features=1000)
    X = tfidf.fit_transform(pre_classified_df["corpus"])
    # 计算不同簇数下的SSE
    inertia_values = []
    k_values = range(1, pre_cluster_num + 1)
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=1)
        kmeans.fit(X)
        inertia_values.append(kmeans.inertia_)  # inertia_表示SSE
    
    elbow_k_residual, coef = detect_elbow_k_residual(k_values, inertia_values)
    print(f"基于残差分析检测到的拐点k={elbow_k_residual}对应的SSE={polynomial(elbow_k_residual, *coef)}")

    # 可视化结果
    plt.plot(k_values, inertia_values, 'bo-', label='raw data')
    plt.plot(k_values, polynomial(k_values, *coef), 'r--', label='fitting curve')
    plt.plot(elbow_k_residual, polynomial(elbow_k_residual, *coef), 'go', label='Detected turning point')
    plt.xlabel('n_clusters (k)')
    plt.ylabel('SSE')
    plt.legend()
    plt.show()

    # KMeans聚类
    n_clusters = elbow_k_residual
    kmeans = KMeans(n_clusters=n_clusters, random_state=1)
    pre_classified_df['cluster'] = kmeans.fit_predict(X)

    pre_classified_summary = get_pre_classified_summary(kmeans, tfidf, pre_classified_df, n_clusters, content_col, sentiment_col)

    return pre_classified_df, pre_classified_summary


def get_classified_summary(classified_df, sentiment_col, classified_col):
    classified_df = classified_df.copy()
    classified_summary = classified_df.groupby(by=classified_col)[[classified_col]].agg({classified_col: 'count'}).rename(
        columns={classified_col: 'count'})
    classified_summary['percentage'] = classified_summary['count'] / len(classified_df)
    if sentiment_col is not None:
        classified_df[sentiment_col] = classified_df[sentiment_col].astype('float64')
        classified_summary[sentiment_col] = classified_df.groupby(by=classified_col)[sentiment_col].mean()
    classified_summary = classified_summary.reset_index()
    classified_summary['推理原因'] = ''
    return classified_df, classified_summary
