#######################################
# Hands-On Machine Learning
# Chapter 2) End2End ML project
#######################################

# Packages
import re
import sys
import sklearn
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

assert sklearn.__version__ >= "0.20"        # sci-learn version check
assert sys.version_info >= (3, 5)           # python version check
import os
import tarfile
import urllib.request
import pandas as pd
from zlib import crc32
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

# Graph Setting
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Image save path
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "2_end2end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("그림 저장:", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

###### 1) Load the dataset
# DOWNLOAD_ROOT = "https://raw.githubusercontent.com/rickiepark/handson-ml2/master/"
# HOUSING_PATH = os.path.join("datasets", "housing")
# HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
#
# def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
#     if not os.path.isdir(housing_path):
#         os.makedirs(housing_path)
#     tgz_path = os.path.join(housing_path, "housing.tgz")
#     urllib.request.urlretrieve(housing_url, tgz_path)
#     housing_tgz = tarfile.open(tgz_path)
#     housing_tgz.extractall(path=housing_path)
#     housing_tgz.close()
#
# fetch_housing_data()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()
# housing.head()                              # 맨 앞 #개의 데이터
# housing.info()                              # 인덱스와 열 정보
# housing['ocean_proximity'].value_counts()   # unique와 개수
# housing.describe()                          # min max 등 수리통계량

# # Simple visualization
# housing.hist(bins=50, figsize=(20,15))
# save_fig("attribute_histogram_plots")
# plt.show()

# Set random_seed
SEED = 42
np.random.seed(SEED)

# 기본적인 train test 분리
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)

# 해시 값을 통한 train_test 분할
# (id로 추가되는 data도 자동적으로 분할, 중첩이 없도록 --> 책 참고)

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

housing_with_id = housing.reset_index()
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

# 새 데이터가 항상 데이터셋의 끝에 추가되지 않고, 행이 변경될 수 있는 경우
# 고유 식별자를 만들 때 안전한 특성을 사용해야함
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

# sklearn 함수를 사용해 train test 분할하는 방법 (권장)
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=SEED)
test_set.head()

# median_income을 income_cat이라는 범주형 열로 변환
housing["median_income"].hist()
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
housing["income_cat"].value_counts()
housing["income_cat"].hist()

# Stratified Shuffle 계층 샘플링 (전체 데이터셋의 한 특성 비율과 동일한 sampling)
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

def income_cat_proportions(data):
    return data['income_cat'].value_counts() / len(data)

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

compare_props = pd.DataFrame({
    "Overall": income_cat_proportions(housing),
    "Stratified": income_cat_proportions(strat_test_set),
    "Random": income_cat_proportions(test_set),
    }).sort_index()
compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100


# income_cat 변수 제거
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# 2.4 데이터 이해를 위한 탐색과 시각화
housing = strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude")
# save_fig('bad_visualization_plot')

housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)
# save_fig('better_visualization_plot')

housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
             s=housing['population']/100, label='population', figsize=(10, 7),
             c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True,
             sharex=False)
plt.legend()
# save_fig('housing_prices_scatterplot')

# 지도사진 추가 등의 세밀한 시각화 조정은 ml2 github에 있음
pass

# 상관관계 조사
corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)

from pandas.plotting import scatter_matrix
attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
scatter_matrix(housing[attributes], figsize=(12, 8))
# save_fig('scatter_matrix_plot')

housing.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.1)
plt.axis([0, 16, 0, 550000])
# save_fig("income_vs_house_value_scatterplot")

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

# 2.5 머신러닝 알고리즘을 위한 데이터 준비
housing = strat_train_set.drop('median_house_value', axis=1)
housing_labels = strat_train_set['median_house_value'].copy()

# 2.5.1 데이터 정제
# housing.dropna(subset=['total_bedrooms'])               # 옵션 1
# housing.drop('total_bedrooms', axis=1)                  # 옵션 2
# median = housing['total_bedrooms'].median()             # 옵션 3
# housing['total_bedrooms'].fillna(median, inplace=True)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = 'median')        # 수치형만 가능
housing_num = housing.drop('ocean_proximity', axis=1)

imputer.fit(housing_num)

imputer.statistics_
housing_num.median().values

X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing_num.index)

# 2.5.2 텍스트와 범주형 특성 다루기
housing_cat = housing[['ocean_proximity']]
housing_cat.head(10)
#
# # ordinal encoding
# from sklearn.preprocessing import OrdinalEncoder
# ordinal_encoder = OrdinalEncoder()
# housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
#
# housing_cat_encoded[:10]
# ordinal_encoder.categories_

# one-hot-encoding
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot
housing_cat_1hot.toarray()
cat_encoder.categories_

# 2.5.3 나만의 변환기
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):   # *args나 **kargs가 아닙니다.
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        print(f"fit 함수 호출됨")
        return self     # 더 할일이 없습니다.
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

# 2.5.4 특성 스케일링
# normalization : MinMaxScaler
# standardization : StandardScaler

# 2.5.5 변환 파이프라인
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

housing_num_tr = num_pipeline.fit_transform(housing_num)

# housing 데이터의 모든 열에 대해 파이프라인 적용
from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', OneHotEncoder(), cat_attribs)
])

housing_prepared = full_pipeline.fit_transform(housing)

# 2.6 모델 선택과 훈련
# 2.6.1 훈련 세트에서 훈련하고 평가하기
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

tree_pred = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, tree_pred)
tree_rmse = np.sqrt(tree_mse)
tree_rmse   # 0?!?!?

# 2.6.2 교차 검증을 사용한 평가

