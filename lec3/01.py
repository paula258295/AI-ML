import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram

penguins = pd.read_csv('penguins.csv')
features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
print(penguins['species'].value_counts())
print(penguins[features].isna().sum())

# 02. Usuń brakujące pomiary i wystandaryzuj cztery cechy. species zachowaj wyłącznie do oceny wyników.
data = penguins.dropna(subset=features).copy()
X = StandardScaler().fit_transform(data[features])
print(X.shape)


# Zobacz faktyczne gatunki na wykresie dwóch pomiarów. Pamiętaj: modele wykorzystają 4 cechy, nie tylko te na osiach.
sns.scatterplot(data=data, x='bill_length_mm', y='flipper_length_mm', hue='species')
plt.show()



# 2 · K-means
# 04. Uruchom K-means z losową inicjalizacją dla k=3. Sprawdź wynik na wykresie: kolor = gatunek, kształt = klaster.

km = KMeans(n_clusters=3, init='random', n_init=1, random_state=42)
data['kmeans'] = km.fit_predict(X).astype(str)
sns.scatterplot(data=data, x='bill_length_mm', y='flipper_length_mm', hue='species', style='kmeans')
plt.show()



# 05. Dobierz liczbę klastrów: metoda łokcia i silhouette score (k=2…7).

ks = range(2, 8)
inertia = []
silhouette = []
for k in ks:
    model = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    inertia.append(model.inertia_)
    silhouette.append(silhouette_score(X, model.labels_))

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].plot(list(ks), inertia, marker='o')
axes[0].set(title='Metoda łokcia', xlabel='k', ylabel='Inertia')
axes[1].plot(list(ks), silhouette, marker='o')
axes[1].set(title='Silhouette', xlabel='k', ylabel='Score')
plt.tight_layout()
plt.show()



# 3 · K-means++
# 06. Zmień wyłącznie inicjalizację na k-means++. Porównaj inertia i gatunki w klastrach. n_init=1 pozwala porównać pojedyncze uruchomienia.

km_plus = KMeans(n_clusters=3, init='k-means++', n_init=1, random_state=42)
data['kmeans++'] = km_plus.fit_predict(X).astype(str)
print('K-means:', round(km.inertia_, 2))
print('K-means++:', round(km_plus.inertia_, 2))
print(pd.crosstab(data['species'], data['kmeans++']))


# 07. Porównaj na wykresach oba podziały. Gatunek jest oznaczony kolorem, klaster — kształtem.

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, column in zip(axes, ['kmeans', 'kmeans++']):
    sns.scatterplot(data=data, x='bill_length_mm', y='flipper_length_mm',
                    hue='species', style=column, ax=ax)
    ax.set_title(column)
plt.tight_layout()
plt.show()



# 4 · HCA: cztery sposoby łączenia
# Porównamy single (najbliższe punkty), complete (najdalsze), average (średnia odległość) i ward (wzrost wariancji). Wszystkie metody dostają te same standaryzowane dane.

# 08. Narysuj cztery dendrogramy. Dla czytelności użyjemy próby 45 obserwacji (po 15 z każdego gatunku); gatunek nie jest wykorzystywany w samym HCA.

sample = data.groupby('species').sample(n=15, random_state=42)
X_sample = StandardScaler().fit(data[features]).transform(sample[features])
methods = ['single', 'complete', 'average', 'ward']

fig, axes = plt.subplots(2, 2, figsize=(12, 7))
for ax, method in zip(axes.flat, methods):
    dendrogram(linkage(X_sample, method=method), ax=ax, no_labels=True)
    ax.set(title=method, ylabel='Odległość łączenia')
plt.tight_layout()
plt.show()
     

# 09. Uruchom cztery warianty HCA na wszystkich obserwacjach, zawsze z 3 klastrami. Pokaż wyniki na tych samych osiach. Kolor = klaster, kształt = rzeczywisty gatunek.

hca_results = {}
for method in methods:
    labels = AgglomerativeClustering(n_clusters=3, linkage=method).fit_predict(X)
    data[method] = labels.astype(str)
    hca_results[method] = labels

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, method in zip(axes.flat, methods):
    sns.scatterplot(data=data, x='bill_length_mm', y='flipper_length_mm',
                    hue=method, style='species', ax=ax, legend=(method == 'single'))
    ax.set_title(method)
plt.tight_layout()
plt.show()



# 10. Porównaj silhouette score oraz tablice gatunek × klaster dla metod HCA. Etykiety klastrów (0, 1, 2) nie mają tej samej interpretacji między metodami.

scores = {'K-means': silhouette_score(X, km.labels_),
          'K-means++': silhouette_score(X, km_plus.labels_)}
for method, labels in hca_results.items():
    scores[method] = silhouette_score(X, labels)
print(pd.Series(scores).round(3).sort_values(ascending=False))

for method in methods:
    print('\n', method)
    print(pd.crosstab(data['species'], data[method]))

#  single       0.469
#  K-means      0.459
#  ward         0.454
#  complete     0.452
#  K-means++    0.447
#  average      0.442
#  dtype: float64

#  single
# single       0  1    2
# species               
# Adelie     151  0    0
# Chinstrap   67  1    0
# Gentoo       0  0  123

#  complete
# complete     0    1   2
# species                
# Adelie       0  151   0
# Chinstrap    0   14  54
# Gentoo     123    0   0

#  average
# average      0    1  2
# species               
# Adelie     151    0  0
# Chinstrap   68    0  0
# Gentoo       0  119  4

#  ward
# ward         0    1   2
# species                
# Adelie     151    0   0
# Chinstrap   11    0  57
# Gentoo       0  123   0




# Który gatunek jest najłatwiejszy do wydzielenia? Po czym to widać?
# Gatunek Gentoo jest najłatwiejszy do wydzielenia, ponieważ wszystkie jego obserwacje 
# są przypisane do tego samego klastra (klastr 2), co oznacza, że jest on bardziej jednorodny niż inne gatunki.

# Czy K-means++ zawsze daje niższe inertia niż losowa inicjalizacja w pojedynczym uruchomieniu?
# Nie, K-means++ nie zawsze daje niższe inertia niż losowa inicjalizacja w pojedynczym uruchomieniu. 
# K-means++ jest algorytmem inicjalizacji, który ma na celu lepsze rozmieszczenie początkowych centroidów, 
# co często prowadzi do lepszych wyników (niższej inertia) w porównaniu do losowej inicjalizacji. 
# Jednakże, w niektórych przypadkach, losowa inicjalizacja może przypadkowo wybrać centroidy, 
# które prowadzą do równie dobrego lub nawet lepszego wyniku niż K-means++. Dlatego wyniki mogą się 
# różnić w zależności od konkretnego zestawu danych i wybranych punktów początkowych.

# Jakie różnice między single, complete, average i ward widać na dendrogramach i wykresach?
# Różnice między metodami single, complete, average i ward widać zarówno na dendrogramach, jak i na wykresach.
# Na dendrogramach, różne metody łączenia prowadzą do różnych struktur hierarchicznych, co wpływa na sposób 
# grupowania obserwacji. Na przykład, metoda single łączy najbliższe punkty, co może prowadzić do tworzenia 
# długich, cienkich klastrów, podczas gdy metoda complete łączy najdalsze punkty, co może prowadzić do bardziej 
# zwartych klastrów. Metoda average bierze pod uwagę średnią odległość między punktami w klastrach, 
# co może prowadzić do bardziej zrównoważonych klastrów. Metoda ward minimalizuje wzrost wariancji 
# wewnątrz klastrów, co często prowadzi do bardziej jednorodnych i kompaktowych klastrów.

# Czy najwyższy silhouette score musi oznaczać największą zgodność z gatunkami?
# Nie, najwyższy silhouette score nie musi oznaczać największej zgodności z gatunkami.
# Silhouette score mierzy, jak dobrze punkty są przypisane do swoich klastrów w kontekście 
# odległości między klastrami, ale nie uwzględnia rzeczywistych etykiet gatunków.
