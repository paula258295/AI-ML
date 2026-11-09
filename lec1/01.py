import pandas as pd
import matplotlib.pyplot as plt

# TODO: wczytaj plik CSV
df = pd.read_csv("penguins.csv")

# TODO: wyświetl pierwsze 5 wierszy
five_first_lines = df.head()
print('Five first lines: ')
print(five_first_lines)

# TODO: rozmiar tabeli
size = df.shape
print('size of the table: ', size)

# TODO: typy danych i braki
df.info()

# TODO: statystyki opisowe
description = df.describe()
print('Descriptive statistics: ', description)


# TODO: liczba braków w każdej kolumnie
na_in_each_column = df.isna().sum()
print('Number of missing values in each column: ', na_in_each_column)

# TODO: policz duplikaty
print("number of duplicates: ", df.duplicated().sum())

# TODO: usuń powtarzające się wiersze
df = df.drop_duplicates()

# TODO: sprawdź rozmiar po czyszczeniu
print('size of the table after cleaning: ', df.shape)


# TODO: jakie wartości występują w kolumnie sex?
print('Values in the column sex: ')
print(df["sex"].value_counts(dropna=False))

# TODO: usuń spacje i zamień litery na małe
df["sex"] = df["sex"].str.strip().str.lower()

# TODO: sprawdź wynik
print("After cleaning: ")
print(df["sex"].value_counts(dropna=False))


# TODO: maksimum masy
print("Maximum body mass: ", df["body_mass_g"].max())

# TODO: ustaw NaN dla masy > 8000 g
df.loc[df["body_mass_g"] > 8000, "body_mass_g"] = float("nan")

# TODO: sprawdź maksimum jeszcze raz
print("Maximum body mass after cleaning: ", df["body_mass_g"].max())


# TODO: mediana masy
df["body_mass_g"] = df["body_mass_g"].fillna(df["body_mass_g"].median())

# TODO: mediana długości dzioba
df["bill_length_mm"] = df["bill_length_mm"].fillna(df["bill_length_mm"].median())

# TODO: brak płci -> "unknown"
df["sex"] = df["sex"].fillna("unknown")

# TODO: kontrola pozostałych braków
print("Remaining missing values:")
print(df.isna().sum())


# TODO: liczebność gatunków
print("Species counts:")
print(df["species"].value_counts())


# TODO: średnia masa według species
print("Average body mass by species: ")
print(df.groupby("species")["body_mass_g"].mean().round(1))

# TODO: histogram masy (np. 15 przedziałów)
ax = df["body_mass_g"].hist(bins=15, edgecolor="white", figsize=(7, 4))
ax.set(title="Distribution of penguin body mass", xlabel="Body mass [g]", ylabel="Number of observations")
plt.tight_layout()
plt.show()


# TODO: liczba pozostałych braków
print("Missings: ", df.isna().sum().sum())

# TODO: liczba duplikatów
print("Duplicates: ", df.duplicated().sum())

# TODO: zapisz CSV bez indeksu
df.to_csv("penguins_clean.csv", index=False)
print("Saved")




# Dlaczego mediana może być lepsza od średniej przy uzupełnianiu braków?


# Dlaczego nie każdą wartość odstającą należy automatycznie usuwać?


# Jakie informacje o gatunkach widać w value_counts() i groupby()?



