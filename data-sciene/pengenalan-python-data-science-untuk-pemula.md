---
description: Rangkuman hasil belajar dari Kaggle
layout: landing
---

# Pengenalan Python Data Science untuk Pemula

{% hint style="info" %}
Kode-kode di bawah ditulis dan dijalankan mengunakan Jupyter Notebook
{% endhint %}

### Pemanasan dulu gan!

```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
```

```python
data = pd.read_csv('pokemon.csv')
data.head()
```

|   | # | Name          | Type 1 | Type 2 | HP | Attack | Defense | Sp. Atk | Sp. Def | Speed | Generation | Legendary |
| - | - | ------------- | ------ | ------ | -- | ------ | ------- | ------- | ------- | ----- | ---------- | --------- |
| 0 | 1 | Bulbasaur     | Grass  | Poison | 45 | 49     | 49      | 65      | 65      | 45    | 1          | False     |
| 1 | 2 | Ivysaur       | Grass  | Poison | 60 | 62     | 63      | 80      | 80      | 60    | 1          | False     |
| 2 | 3 | Venusaur      | Grass  | Poison | 80 | 82     | 83      | 100     | 100     | 80    | 1          | False     |
| 3 | 4 | Mega Venusaur | Grass  | Poison | 80 | 100    | 123     | 122     | 120     | 80    | 1          | False     |
| 4 | 5 | Charmander    | Fire   | NaN    | 39 | 52     | 43      | 60      | 50      | 65    | 1          | False     |

```python
data.info()
```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 800 entries, 0 to 799
Data columns (total 12 columns):
 #   Column      Non-Null Count  Dtype 
---  ------      --------------  ----- 
 0   #           800 non-null    int64 
 1   Name        799 non-null    object
 2   Type 1      800 non-null    object
 3   Type 2      414 non-null    object
 4   HP          800 non-null    int64 
 5   Attack      800 non-null    int64 
 6   Defense     800 non-null    int64 
 7   Sp. Atk     800 non-null    int64 
 8   Sp. Def     800 non-null    int64 
 9   Speed       800 non-null    int64 
 10  Generation  800 non-null    int64 
 11  Legendary   800 non-null    bool  
dtypes: bool(1), int64(8), object(3)
memory usage: 69.7+ KB
```

```python
# Hubungan antarvariabel
data.corr()
```

|            | #        | HP       | Attack   | Defense  | Sp. Atk  | Sp. Def  | Speed     | Generation | Legendary |
| ---------- | -------- | -------- | -------- | -------- | -------- | -------- | --------- | ---------- | --------- |
| #          | 1.000000 | 0.097712 | 0.102664 | 0.094691 | 0.089199 | 0.085596 | 0.012181  | 0.983428   | 0.154336  |
| HP         | 0.097712 | 1.000000 | 0.422386 | 0.239622 | 0.362380 | 0.378718 | 0.175952  | 0.058683   | 0.273620  |
| Attack     | 0.102664 | 0.422386 | 1.000000 | 0.438687 | 0.396362 | 0.263990 | 0.381240  | 0.051451   | 0.345408  |
| Defense    | 0.094691 | 0.239622 | 0.438687 | 1.000000 | 0.223549 | 0.510747 | 0.015227  | 0.042419   | 0.246377  |
| Sp. Atk    | 0.089199 | 0.362380 | 0.396362 | 0.223549 | 1.000000 | 0.506121 | 0.473018  | 0.036437   | 0.448907  |
| Sp. Def    | 0.085596 | 0.378718 | 0.263990 | 0.510747 | 0.506121 | 1.000000 | 0.259133  | 0.028486   | 0.363937  |
| Speed      | 0.012181 | 0.175952 | 0.381240 | 0.015227 | 0.473018 | 0.259133 | 1.000000  | -0.023121  | 0.326715  |
| Generation | 0.983428 | 0.058683 | 0.051451 | 0.042419 | 0.036437 | 0.028486 | -0.023121 | 1.000000   | 0.079794  |
| Legendary  | 0.154336 | 0.273620 | 0.345408 | 0.246377 | 0.448907 | 0.363937 | 0.326715  | 0.079794   | 1.000000  |

```python
# Bikin Correlation Map, biar tau hubungan antarvariabelnya
f, ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(),annot=True, linewidths= .5, fmt='.1f', ax=ax)
plt.show()
```

![](../.gitbook/assets/output\_7\_0.png)

```python
data.head(10)
```

|   | #  | Name             | Type 1 | Type 2 | HP | Attack | Defense | Sp. Atk | Sp. Def | Speed | Generation | Legendary |
| - | -- | ---------------- | ------ | ------ | -- | ------ | ------- | ------- | ------- | ----- | ---------- | --------- |
| 0 | 1  | Bulbasaur        | Grass  | Poison | 45 | 49     | 49      | 65      | 65      | 45    | 1          | False     |
| 1 | 2  | Ivysaur          | Grass  | Poison | 60 | 62     | 63      | 80      | 80      | 60    | 1          | False     |
| 2 | 3  | Venusaur         | Grass  | Poison | 80 | 82     | 83      | 100     | 100     | 80    | 1          | False     |
| 3 | 4  | Mega Venusaur    | Grass  | Poison | 80 | 100    | 123     | 122     | 120     | 80    | 1          | False     |
| 4 | 5  | Charmander       | Fire   | NaN    | 39 | 52     | 43      | 60      | 50      | 65    | 1          | False     |
| 5 | 6  | Charmeleon       | Fire   | NaN    | 58 | 64     | 58      | 80      | 65      | 80    | 1          | False     |
| 6 | 7  | Charizard        | Fire   | Flying | 78 | 84     | 78      | 109     | 85      | 100   | 1          | False     |
| 7 | 8  | Mega Charizard X | Fire   | Dragon | 78 | 130    | 111     | 130     | 85      | 100   | 1          | False     |
| 8 | 9  | Mega Charizard Y | Fire   | Flying | 78 | 104    | 78      | 159     | 115     | 100   | 1          | False     |
| 9 | 10 | Squirtle         | Water  | NaN    | 44 | 48     | 65      | 50      | 64      | 43    | 1          | False     |

```python
data.columns
```

```
Index(['#', 'Name', 'Type 1', 'Type 2', 'HP', 'Attack', 'Defense', 'Sp. Atk',
       'Sp. Def', 'Speed', 'Generation', 'Legendary'],
      dtype='object')
```

## PENGENALAN PYTHON

#### MATPLOTLIB

* Line plot bagus kalau sumbu x-nya waktu
* Scatter bagus kalau ada korelasi dua variabel
* Histogram bagus kalau butuh lihat distribusi data numerik
* Yang bisa diatur-atur: warna, label, ketebalan garis, judul, opacity, grid, figsize, ketebalan sumbu, dan gaya sumbu.

```python
# Line plot

data.Speed.plot(kind = 'line', color = 'r', label = 'Speed', linewidth = 1, alpha = .5, grid = True, linestyle = ':')
data.Defense.plot(color = 'b', label = 'Defense', linewidth = 1, alpha = .5, grid = True, linestyle = '-.')
### color = warna, label = label, linewidth = tebal garis, alpha = opacity, grid = grid, linestyle = gaya garis

plt.legend(loc = 'upper right')
# legend = label di dalam grafik

plt.xlabel('x axis')
plt.ylabel('y axis')
# label = nama label

plt.title('Line Plot')
# title = nama grafik

plt.show()
```

![](../.gitbook/assets/output\_13\_0.png)

```python
# Scatter plot

data.plot(kind = 'scatter', x = 'Attack', y = 'Defense', alpha = .5, color = 'b')
plt.xlabel('Attack')
plt.ylabel('Defense')
plt.title('Attack Defense Pokemon')
```

![](../.gitbook/assets/output\_14\_1.png)

```
Text(0.5, 1.0, 'Attack Defense Pokemon')


```

```python
# Histogram

data.Speed.plot(kind = 'hist', bins = 50, figsize = (15, 10))
# bins = jumlah batang dalam grafik

plt.show()
```

![](../.gitbook/assets/output\_15\_0.png)

```python
# clf() = membersihkan
data.Speed.plot(kind = 'hist', bins = 50)
plt.clf()
```

```
<Figure size 432x288 with 0 Axes>
```

#### DICTIONARY

* Punya 'key' sama 'value'. Contoh: dictionary = {'Indonesia': 'Jakarta'}
* Key-nya Indonesia
* Values-nya Jakarta
* Lebih cepet dari pada lists

```python
kamus = {'Indo': 'Jakarta', 'Malay': 'KL', 'Ph': 'Manila'}
print(kamus.keys())
print(kamus.values())
```

```
dict_keys(['Indo', 'Malay', 'Ph'])
dict_values(['Jakarta', 'KL', 'Manila'])
```

```python
# Keys bisa pakai string, boolean, float, integer, tubles
# List tidak kekal
# Keys itu unik

# Memperbarui entri 
kamus['Indo'] = "Nusantara"
print(kamus)

# Menambah entri
kamus['Singa'] = "Singapura"
print(kamus)

# Menghapus entri
del kamus['Ph']
print(kamus)

# Memastikan ada tidaknya
print('Singa' in kamus)

# Menghapus semua entri
kamus.clear()

print(kamus)
```

```
{'Indo': 'Nusantara', 'Malay': 'KL', 'Ph': 'Manila'}
{'Indo': 'Nusantara', 'Malay': 'KL', 'Ph': 'Manila', 'Singa': 'Singapura'}
{'Indo': 'Nusantara', 'Malay': 'KL', 'Singa': 'Singapura'}
True
{}
```

#### PANDAS

```python
# Membaca data format CSV
data = pd.read_csv('pokemon.csv')
```

```python
series = data['Defense']
print(type(series))

data_frame = data[['Defense']]
print(type(data_frame))
```

```
<class 'pandas.core.series.Series'>
<class 'pandas.core.frame.DataFrame'>
```

#### Logika, control flow, dan filtering

* Operasi pembanding: ==, >, <, <=, >=, !=
* Operasi boolean: and, not, or
* Filtering pandas

```python
# Pembanding
print(3 != 2)
print(3 > 4)
print(0 == 0)

# Boolean
print(True and False)
print(False or True)
print(not False)
```

```
True
False
True
False
True
True
```

```python
# 1 - Filtering pandas data frame
x = data['Defense']>200
# cuma ada 3 pokemon yang punya nilai defense lebih dari 200 poin

# Menampilkan data
data[x]
```

|     | #   | Name         | Type 1 | Type 2 | HP | Attack | Defense | Sp. Atk | Sp. Def | Speed | Generation | Legendary |
| --- | --- | ------------ | ------ | ------ | -- | ------ | ------- | ------- | ------- | ----- | ---------- | --------- |
| 224 | 225 | Mega Steelix | Steel  | Ground | 75 | 125    | 230     | 55      | 95      | 30    | 2          | False     |
| 230 | 231 | Shuckle      | Bug    | Rock   | 20 | 10     | 230     | 10      | 230     | 5     | 2          | False     |
| 333 | 334 | Mega Aggron  | Steel  | NaN    | 70 | 140    | 230     | 60      | 80      | 50    | 3          | False     |

```python
# 2 - Filtering pandas dengan logical_and
data[np.logical_and(data['Defense']>200, data['Attack']>100)]
# Hanya ada 2 pokemon yang memiliki defense di atas 200 poin dan attack di atas 100 poin
```

|     | #   | Name         | Type 1 | Type 2 | HP | Attack | Defense | Sp. Atk | Sp. Def | Speed | Generation | Legendary |
| --- | --- | ------------ | ------ | ------ | -- | ------ | ------- | ------- | ------- | ----- | ---------- | --------- |
| 224 | 225 | Mega Steelix | Steel  | Ground | 75 | 125    | 230     | 55      | 95      | 30    | 2          | False     |
| 333 | 334 | Mega Aggron  | Steel  | NaN    | 70 | 140    | 230     | 60      | 80      | 50    | 3          | False     |

```python
# Sama seperti atas cuma beda pakai simbol &
data[(data['Defense']>180) & (data['Attack']>90)]
```

|     | #   | Name         | Type 1 | Type 2 | HP | Attack | Defense | Sp. Atk | Sp. Def | Speed | Generation | Legendary |
| --- | --- | ------------ | ------ | ------ | -- | ------ | ------- | ------- | ------- | ----- | ---------- | --------- |
| 224 | 225 | Mega Steelix | Steel  | Ground | 75 | 125    | 230     | 55      | 95      | 30    | 2          | False     |
| 333 | 334 | Mega Aggron  | Steel  | NaN    | 70 | 140    | 230     | 60      | 80      | 50    | 3          | False     |
| 414 | 415 | Regirock     | Rock   | NaN    | 80 | 100    | 200     | 50      | 100     | 50    | 3          | True      |
| 789 | 790 | Avalugg      | Ice    | NaN    | 95 | 117    | 184     | 44      | 46      | 28    | 6          | False     |

#### WHILE dan FOR LOOPS

```python
# Tetap dalam kondisi loof if (selama i tidak sama dengan 5) is true 
i = 0
while i != 5:
    print('i adalah: ', i)
    i += 1

print(i, ' sama dengan 5')
```

```
i adalah:  0
i adalah:  1
i adalah:  2
i adalah:  3
i adalah:  4
5  sama dengan 5
```

```python
# Mirip seperti yang atas, cuma dalam bentuk list 
daftar = [1, 2, 3, 4, 5]
for i in daftar:
    print('i adalah ', i)
print('')

# Enumerasi index dan nilai dari list
for index, value in enumerate(daftar):
    print(index, " : ", value) # index : nilai = 0 : 1, 1 : 2, 2: 3, 3 : 4
print('')

# Untuk kamus
kamus = {'Indo': 'Jakarta', 'Malay': 'KL', 'Ph': 'Manila'}
for key, value in kamus.items():
    print(key, " : ", value)
print('')

# Untuk pandas bisa peroleh index dan value
for index, value in data[['Attack']][0:1].iterrows():
    print(index, " : ", value)
```

```
i adalah  1
i adalah  2
i adalah  3
i adalah  4
i adalah  5

0  :  1
1  :  2
2  :  3
3  :  4
4  :  5

Indo  :  Jakarta
Malay  :  KL
Ph  :  Manila

0  :  Attack    49
Name: 0, dtype: int64
```
