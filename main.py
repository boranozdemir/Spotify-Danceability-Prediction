###### SPOTIFY: DANCEABILITY PROJECT #####



import pandas as pd

pd.set_option("display.max_columns", None)

df = pd.read_csv("dataset/dataset.csv")

#####################################################General Look at Dataset###########################################

df.info()


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)

df = df.drop("Unnamed: 0", axis=1)

df[df["artists"].isnull()]


df = df.drop(65900, axis=0)


########################################### Getting column names for each category.####################################


def grab_col_names(dataframe, cat_th=13, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
        Numerik değişken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi

    Notes
    ------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]

    num_but_cat = [col for col in df.columns if df[col].nunique() < cat_th and df[col].dtypes in ["int", "float"]]

    cat_but_car = [col for col in df.columns if
                   df[col].nunique() > car_th and str(df[col].dtypes) in ["category", "object"]]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)



# Observations: 113999
# Variables: 20
# cat_cols: 3
# num_cols: 12
# cat_but_car: 5
# num_but_cat: 2


# cat_cols:
# ['explicit', 'mode','key', 'time_signature']

# num_cols:
# ['popularity', 'duration_ms', 'danceability', 'energy',  'loudness',
# 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# cat_but_car:
# ['track_id', 'artists', 'album_name', 'track_name', 'track_genre']


#############################################
# 2. Analysis of Categorical Variables
#############################################

import seaborn as sns

import matplotlib.pyplot as plt

df["explicit"] = df["explicit"].astype(int)


def cat_summary(dataframe, col_name, plot=False):
    print(col_name)
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


for col in cat_cols:
    cat_summary(df, col)



#############################################
# 3. Analysis of Numerical Variables
#############################################

outcome = "danceability"

num_cols.remove("danceability")


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in num_cols:
    num_summary(df, col)

#############################################
# 4. Analysis of Target Variable
#############################################


def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


for col in cat_cols:
    target_summary_with_cat(df, outcome, col)



##################### Key/Danceability ############################

df['dance_prof'] = df['danceability'].apply(lambda x: 1 if x > 0.5 else 0)
grouped = df.groupby(['dance_prof', 'key']).size().unstack(fill_value=0)
"""
grouped
key           0     1     2     3     4   ...    7     8     9     10    11
dance_prof                                ...                              
0           4412  3364  4391  1303  3159  ...  4401  2240  3971  2276  2627
1           8649  7408  7253  2267  5849  ...  8843  5120  7342  5180  6655
[2 rows x 12 columns]
pd.set_option("display.max_columns", None)
grouped
key           0     1     2     3     4     5     6     7     8     9     10  \
dance_prof                                                                     
0           4412  3364  4391  1303  3159  3085  2348  4401  2240  3971  2276   
1           8649  7408  7253  2267  5849  6283  5573  8843  5120  7342  5180   
key           11  
dance_prof        
0           2627  
1           6655 
"""

df['dance_prof'] = df['danceability'].apply(lambda x: 1 if x > 0.7 else 0)
grouped = df.groupby(['dance_prof', 'key']).size().unstack(fill_value=0)

"""
key            0     1     2     3     4     5     6      7     8     9   \
dance_prof                                                                 
0           10121  7715  9416  2820  7077  7264  5794  10069  5384  8928   
1            2940  3057  2228   750  1931  2104  2127   3175  1976  2385   
key           10    11  
dance_prof              
0           5446  6719  
1           2010  2563  
"""

genres = []

for i in dict.values():
    for j in i:
        genres.append(j)


#############################################
# 5. Analysis of Correlation
#############################################


corr = df[num_cols].corr()

print(corr)


sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()
plt.close()

#############################################
# 6. Data Engineering & Feature Engineering
#############################################

# Tresholds


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        print(f"{col_name} = True")
        return True
    else:
        print(f"{col_name} = False")
        return False





for col in num_cols:
    check_outlier(df, col)


for col in num_cols:
    replace_with_thresholds(df, col)



for col in num_cols:
    check_outlier(df, col)



