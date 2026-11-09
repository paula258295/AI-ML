import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sns.set_theme(style="whitegrid", palette="deep")
pd.set_option("display.max_columns", 20)
df = pd.read_csv("wine.csv")
print(df.head())


# Sprawdź wymiary, typy, brakujące wartości i duplikaty. Następnie policz statystyki opisowe. Nie usuwaj żadnych kolumn bez uzasadnienia.
# TODO: shape, info(), isna().sum(), duplicated().sum(), describe()
print("Shape of the dataset: ", df.shape)
print("Data types and missing values: ")
df.info()
print("Number of missing values in each column: ")
print(df.isna().sum())
print("Number of duplicates: ", df.duplicated().sum())
print("Descriptive statistics: ")
print(df.describe().T.round(2))


print("--------------- 2 ---------------")
# Zobacz rozkład klas, histogram jednej cechy, boxplot innej oraz korelacje. Wskaż parę silnie skorelowanych cech.
# TODO: countplot(wine_class), hist(alcohol), boxplot(x=wine_class, y=proline)
# TODO: heatmap korelacji cech numerycznych (bez wine_class)
fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
print("Class distribution")
sns.countplot(data=df, x="wine_class", ax=ax[0])
print("Histogram of alcohol")
sns.histplot(data=df, x="alcohol", bins=18, ax=ax[1])
print("Boxplot of proline by wine_class")
sns.boxplot(data=df, x="wine_class", y="proline", ax=ax[2])
plt.tight_layout()
plt.show()

print("Correlation heatmap of numerical features")
feature_cols = df.columns.drop(["sample_id", "wine_class"])
plt.figure(figsize=(10, 8))
sns.heatmap(df[feature_cols].corr(), cmap="vlag", center=0, vmin=-1, vmax=1)
plt.title("Correlation heatmap of numerical features")
plt.tight_layout()
plt.show()

print("--------------- 3 ---------------")


# Oddziel cechy od etykiety. Wykonaj standaryzację (StandardScaler). Sprawdź średnie i odchylenia standardowe po transformacji.
# TODO: utwórz X, y; dopasuj StandardScaler; otrzymaj X_scaled
feature_cols = df.columns.drop(["sample_id", "wine_class"])
X = df[feature_cols]
y = df["wine_class"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Average:", np.round(X_scaled.mean(axis=0), 2))
print("Standard deviations:", np.round(X_scaled.std(axis=0), 2))

# Average: [-0. -0. -0. -0. -0.  0. -0.  0. -0.  0.  0.  0. -0.]
# Standard deviations: [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]


# Dopasuj PCA bez ograniczania liczby składowych. Zapisz wartości własne, udział wyjaśnionej wariancji i wariancję skumulowaną.
# TODO: PCA(), fit(X_scaled); eigenvalues, explained_ratio, cumulative
pca_full = PCA().fit(X_scaled)
eigenvalues = pca_full.explained_variance_
explained_ratio = pca_full.explained_variance_ratio_
cumulative = np.cumsum(explained_ratio)
display(pd.DataFrame({
    "PC": np.arange(1, len(eigenvalues) + 1),
    "wartosc_wlasna": eigenvalues,
    "wariancja_%": 100 * explained_ratio,
    "skumulowana_%": 100 * cumulative
}).round(2))