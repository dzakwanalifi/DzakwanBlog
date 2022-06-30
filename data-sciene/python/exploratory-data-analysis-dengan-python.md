---
description: >-
  Hasil rangkuman praktik dari Trial Class Rakamin dalam materi Exploratory Data
  Analysis pada dataset sintetik "Probabilitas Kebotakan".
---

# Exploratory Data Analysis dengan Python

{% hint style="info" %}
Kode-kode di bawah ditulis dan dijalankan mengunakan Google Colaboratory
{% endhint %}

```
from google.colab import drive
```

```python
drive.mount('/content/drive')
```

```
Mounted at /content/drive
```

```python
# Mengimpor library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

#### Mengubah parameter default matplotlib

Parameter default matplotlib dapat diubah dengan rcParams sebagai berikut (opsional: estetika plot).

```python
from matplotlib import rcParams

rcParams['figure.figsize'] = (12, 4)
rcParams['lines.linewidth'] = 3
rcParams['xtick.labelsize'] = 'small'
rcParams['ytick.labelsize'] = 'small'
```

#### Load dataset

Melihat dataset sintetis prediksi kebotakan. Agar running time-nya tidak teralu lama, hanya akan diambil 1000 sampel baris data saja.

```python
df = pd.read_csv('/content/drive/MyDrive/botak.csv').sample(1000, random_state=42)
```

### Descriptive Statistics

Memeriksa kolom-kolom dan nilai yang hilang dengan `df.info()`

```python
df.info()
```

```
<class 'pandas.core.frame.DataFrame'>
Int64Index: 1000 entries, 6255 to 6621
Data columns (total 13 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   umur           1000 non-null   float64
 1   jenis_kelamin  999 non-null    object 
 2   pekerjaan      987 non-null    object 
 3   provinsi       1000 non-null   object 
 4   gaji           1000 non-null   float64
 5   is_menikah     1000 non-null   int64  
 6   is_keturunan   1000 non-null   float64
 7   berat          1000 non-null   float64
 8   tinggi         1000 non-null   float64
 9   sampo          1000 non-null   object 
 10  is_merokok     1000 non-null   int64  
 11  pendidikan     1000 non-null   object 
 12  botak_prob     1000 non-null   float64
dtypes: float64(6), int64(2), object(5)
memory usage: 109.4+ KB
```

Dapat diketahui bahwa:

* Dataframe memiliki total 1000 baris dan 13 kolom
* Dataframe masih memiliki _null_ values di kolom `pekerjaan` dan `jenis_kelamin`
* Target klasifikasi adalah kolom `botak_prob` dengan tipe data `float64`
* Sisanya adalah feature (predictor)

Untuk memudahkan kolom kategorik dan numerik dapat dipisahkan:

```python
kat = ['jenis_kelamin', 'pekerjaan', 'sampo', 'pendidikan', 'provinsi']
num = ['umur', 'gaji', 'is_menikah', 'is_keturunan', 'berat', 'tinggi', 'is_merokok', 'botak_prob']
```

#### Sampling untuk memahami data dengan `df.sample()`

Setelah mengetahui kolom apa saja yang ada dalam dataset, kita lakukan sampling untuk memastikan apakah isi kolomnya sesuai ekspektasi. Biasakan melakukan ini beberapa kali karena seringkali apabila ada keanehan tidak keluar pada sampling pertama.

```python
df.sample(10)
```

|      | umur | jenis\_kelamin | pekerjaan      | provinsi      | gaji         | is\_menikah | is\_keturunan | berat     | tinggi     | sampo           | is\_merokok | pendidikan | botak\_prob |
| ---- | ---- | -------------- | -------------- | ------------- | ------------ | ----------- | ------------- | --------- | ---------- | --------------- | ----------- | ---------- | ----------- |
| 4747 | 39.0 | Perempuan      | Pengangguran   | Kupang        | 5.507331e+06 | 0           | 0.0           | 58.455203 | 164.212718 | Pantone         | 0           | SD         | 0.471786    |
| 7748 | 37.0 | Laki-laki      | PNS            | Pangkalpinang | 8.963102e+06 | 0           | 1.0           | 59.017732 | 159.674614 | Pantone         | 1           | SMP        | 0.741001    |
| 2373 | 47.0 | Laki-laki      | Freelance      | Banjarmasin   | 6.164560e+06 | 0           | 0.0           | 58.451823 | 164.385422 | Shoulder & Head | 1           | SMA        | 0.338584    |
| 2301 | 32.0 | Laki-laki      | PNS            | Mataram       | 6.254391e+06 | 0           | 0.0           | 46.365804 | 153.417912 | Merpati         | 1           | S1         | 0.412617    |
| 6913 | 35.0 | Perempuan      | PNS            | Semarang      | 2.210889e+07 | 0           | 0.0           | 49.549669 | 155.667067 | Merpati         | 1           | S2         | 0.363062    |
| 7871 | 50.0 | Laki-laki      | Freelance      | Mataram       | 1.118140e+07 | 0           | 0.0           | 53.912280 | 157.347171 | Merpati         | 0           | S1         | 0.202815    |
| 2404 | 61.0 | Laki-laki      | Pegawai swasta | Yogyakarta    | 1.176255e+07 | 1           | 1.0           | 46.692613 | 147.338070 | Deadbuoy        | 0           | SMA        | 0.858448    |
| 61   | 46.0 | Laki-laki      | Pegawai swasta | Kendari       | 4.048905e+06 | 0           | 0.0           | 46.311319 | 155.249996 | Merpati         | 0           | S2         | 0.301477    |
| 4024 | 42.0 | Perempuan      | PNS            | Palembang     | 1.068752e+07 | 0           | 0.0           | 86.458696 | 162.655053 | Merpati         | 1           | S1         | 0.408850    |
| 2249 | 37.0 | Laki-laki      | NaN            | Mataram       | 7.936113e+06 | 0           | 0.0           | 66.054809 | 158.470900 | Moonsilk        | 1           | S2         | 0.399189    |

* Target `botak_prob` bertipe float dengan range 0-1, dimana 1 menunjukkan kemungkinan 100%.
* Tidak ada yang aneh dari input setiap kolom.

#### Statistical summary dengan `df.describe()`

Menampilkan ringkasan statistik dataframe, baik untuk kategorikal maupun numerikal. Hal ini dilakukan untuk mengecek keberadan outlier dan karakteristik distribusi untuk `feature` numerik.

```python
# Ringkasan statistik kolom numerik
df[num].describe()
```

|       | umur        | gaji         | is\_menikah | is\_keturunan | berat       | tinggi      | is\_merokok | botak\_prob |
| ----- | ----------- | ------------ | ----------- | ------------- | ----------- | ----------- | ----------- | ----------- |
| count | 1000.000000 | 1.000000e+03 | 1000.000000 | 1000.00000    | 1000.000000 | 1000.000000 | 1000.000000 | 1000.000000 |
| mean  | 40.008000   | 9.223088e+06 | 0.033000    | 0.18000       | 56.162619   | 157.486031  | 0.492000    | 0.390735    |
| std   | 9.886642    | 4.739127e+06 | 0.178726    | 0.38438       | 9.258898    | 6.548078    | 0.500186    | 0.192807    |
| min   | 8.000000    | 1.500000e+06 | 0.000000    | 0.00000       | 40.921334   | 142.554038  | 0.000000    | -0.068044   |
| 25%   | 33.000000   | 5.849544e+06 | 0.000000    | 0.00000       | 50.099311   | 152.899106  | 0.000000    | 0.260230    |
| 50%   | 40.000000   | 8.107307e+06 | 0.000000    | 0.00000       | 54.102392   | 157.293445  | 0.000000    | 0.368024    |
| 75%   | 47.000000   | 1.132830e+07 | 0.000000    | 0.00000       | 60.485423   | 161.493659  | 1.000000    | 0.508568    |
| max   | 72.000000   | 4.179443e+07 | 1.000000    | 1.00000       | 128.643924  | 193.484937  | 1.000000    | 1.000000    |

Dari _statistical summary_ kolom numerik di atas didapati bahwa:

* Distribusi nilai kolom `umur` terlihat normal/simetrik (mean dan median cukup dekat dan nilai Q2 berada tepat di tengah Q1 dan Q3)
* Kolom `gaji` tidak simetrik karena mean > median
* Ada data yang memiliki probabilitas minus pada `botak_prob`. Baris ini harus dihilangkan pada tahap pre-processing.

```python
# Jumlah baris dengan botak_prob bernilai negatif
df[df['botak_prob']<0].shape[0]
```

```
11
```

```python
# Ringkasan statistik kolom kategorik
df[kat].describe()
```

.dataframe tbody tr th:only-of-type { vertical-align: middle; } .dataframe tbody tr th { vertical-align: top; } .dataframe thead th { text-align: right; }

|        | jenis\_kelamin | pekerjaan      | sampo   | pendidikan | provinsi |
| ------ | -------------- | -------------- | ------- | ---------- | -------- |
| count  | 999            | 987            | 1000    | 1000       | 1000     |
| unique | 2              | 4              | 5       | 6          | 34       |
| top    | Laki-laki      | Pegawai swasta | Merpati | S1         | Denpasar |
| freq   | 684            | 409            | 228     | 562        | 41       |

* Mayoritas data pada `jenis_kelamin` adalah laki-laki dengan frekuensi sebanyak 684
* Kolom `provinsi` memiliki jumlah nilai unik sejumlah 34. Memungkinkan tidak berguna untuk dijadikan predictor

### Univariate Analysis

Setelah analisis sederhana dengan statistik deskriptif, sekarang fokus untuk menganalisis kolom satu persatu dengan univariate analysis.

#### Box Plot

Menampilkan grafik box dari data numerik.

```python
for i in range (0, len(num)):
  plt.subplot(1, len(num), i+1)
  sns.boxplot(y=df[num[i]], color='grey', orient='v')
  plt.tight_layout()
```

![](../../.gitbook/assets/eda\_output\_22\_0.png)

Hal yang paling penting diperhatikan dalam box plot adalah outlier.

* Outlier terlihat utamanya pada kolom `gaji`, `berat`, dan `tinggi`
* Kolom `gaji`, `berat`, dan `tinggi` terlihat tidak simetri (skewed) yang ditandai dengan lokasi box yang jauh dari daerah tengah sumbu Y.

#### Distribution Plot

Menampilkan grafik distribusi dari kolom numerik.

```python
plt.figure(figsize=(12,5))
for i in range(0, len(num)):
  plt.subplot(2, len(num)/2, i+1)
  sns.distplot(df[num[i]], color='grey')
  plt.tight_layout()
```

![](../../.gitbook/assets/eda\_output\_25\_1.png)

* Kolom `gaji`, `berat`, dan `tinggi` terlihat skewed seperti pada boxplot
* Kolom lain sudah simetrik distribusinya.

#### Count Plot

Grafik distribusi untuk kolom kategorik.

```python
for i in range(0, len(kat)):
  plt.subplot(2, 3, i+1)
  sns.countplot(df[kat[i]], color='grey', orient='v')
  plt.xticks(rotation=20)
  plt.tight_layout()
```

![](../../.gitbook/assets/eda\_output\_28\_1.png)

* Distribusi `pendidikan` dan `pekerjaan` didominasi oleh 1-2 value.

### Bivariate Analysis

Setelah menganalisis kolom secara individual, berikutnya adalah melihat hubungan antarkolom. Dengan mengetahui hubungan antarkolom dapat membantu untuk memilih feature yang paling penting dan mengesampingkan feature yang redundan.

#### Correlation Heatmap

```python
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(), cmap='Blues', annot=True, fmt='.2f')
```

```
<matplotlib.axes._subplots.AxesSubplot at 0x7fd14b939a90>
```

![](../../.gitbook/assets/eda\_output\_32\_1.png)

* Target `botak_prob` memiliki korelasi positif yang kuat dengan `is_keturunan`
* `botak_prob` memiliki korelasi positif lemah dengan `umur`, `is_menikah`, dan `is_merokok`
* `tinggi` memiliki korelasi positif kuat dengan `berat`. Namun ada kemungkinan kedua featur ini redundan.

### EDA Conclusion

* Data terlihat valid dan tidak ada kecacatan yang major
* Masih ada data dengan target variabel bernilai negatif yang harus di-drop nantinya
* Masih ada data bernilai _NaN_/hilang yang harus diurus pada saat preprocessing
* Ada beberapa distribusi yang sedikit skewed. Hal ini perlu diperhatikan ketika ingin menggunakan model yang memerlukan asumsi distribusi normal
* Beberapa feature memiliki korelasi yang jelas dengan target yang akan dipakai
* Beberapa feature yang tidak berkorelasi akan diabaikan
* Dari Heatmap dapat dilihat: ada feature yang berkoralasi tinggi satu sama lain, namun mungkin hanya akan dipakai salah satu.
