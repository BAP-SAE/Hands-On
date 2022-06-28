# Chapter 1) ML Landscape

#TODO: 1장에 있는 그림을 생성하기위한 코드는 실행해보지 않았음

# Packages
import sys
import os

import numpy as np
import pandas as pd

import sklearn
import sklearn.linear_model

import matplotlib.pyplot as plt
import matplotlib as mpl

# Version Check
assert sys.version_info >= (3, 5), "need to check sys version"
assert sklearn.__version__ >= "0.20", "need to check sklearn version"

# path setting
data_path = os.path.join("datasets", "lifesat", "")

# 데이터 적재
# oecd_bli_without_thousands = pd.read_csv(data_path + "oecd_bli_2015.csv")
oecd_bli = pd.read_csv(data_path + "oecd_bli_2015.csv", thousands=',')

gdp_per_capita = pd.read_csv(data_path + "gdp_per_capita.csv", thousands=',', delimiter='\t',
                             encoding='latin1', na_values="n/a")

# 데이터 준비
def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"] == "TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")

    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)

    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]



country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats['GDP per capita']]
y = np.c_[country_stats['Life satisfaction']]


# Graph setting
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# 데이터 시각화
country_stats.plot(kind='scatter', x='GDP per capita', y='Life satisfaction')
plt.show()

X_new = [[22587]]

# Linear model
lr_model = sklearn.linear_model.LinearRegression()
lr_model.fit(X, y)
print(f"linear_regression: {lr_model.predict(X_new)}")

import sklearn.neighbors
knn_model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)
knn_model.fit(X, y)
print(f"k-nearest neighborhodd: {knn_model.predict(X_new)}")



