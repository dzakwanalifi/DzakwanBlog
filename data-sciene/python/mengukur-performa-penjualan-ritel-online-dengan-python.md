---
description: Hasil latihan homework Trial Class Data Science Rakamin Academy.
---

# Mengukur Performa Penjualan Ritel Online dengan Python

#### Homework Rakamin Trial Class - Mini Case



{% hint style="info" %}
#### Dataset **Attributes**

* Invoice : Nomor invoice 6 digit yang ditetapkan secara unik untuk setiap transaksi. Jika kode ini dimulai dengan huruf 'C', itu menunjukkan pembatalan.
* StockCode : Kode produk (barang). Angka 5 digit yang ditetapkan secara unik untuk setiap produk yang berbeda.
* Description : Nama produk.&#x20;
* Quantity : Jumlah kuantitas setiap produk per transaksi.
* InvoiceDate : Tanggal dan waktu invoice, yakni hari dan waktu saat transaksi dibuat.
* UnitPrice : Harga satuan atau harga produk per unit dalam sterling (£).
* CustomerID : Nomor 5 digit yang ditetapkan secara unik untuk setiap pelanggan.
* Country : Nama negara tempat tinggal pelanggan.

Dataset online\_retail\_II.csv dapat diakses pada tautan [**berikut**](https://drive.google.com/drive/folders/1UbsUuQJgkF-7ilhhNL2tOnpzlJ-WS\_Fu?usp=sharing)****
{% endhint %}

****

### Load Data

```python
# Mengimpor library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pandas.core.indexes.datetimes import DatetimeIndex
```

```python
from google.colab import drive
drive.mount('/content/drive')
```

```
Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
```

```python
df = pd.read_csv('/content/drive/MyDrive/online_retail_II.csv')
df.head()
```

|   | Invoice | StockCode | Description                         | Quantity | InvoiceDate         | Price | Customer ID | Country        |
| - | ------- | --------- | ----------------------------------- | -------- | ------------------- | ----- | ----------- | -------------- |
| 0 | 489434  | 85048     | 15CM CHRISTMAS GLASS BALL 20 LIGHTS | 12       | 2009-12-01 07:45:00 | 6.95  | 13085.0     | United Kingdom |
| 1 | 489434  | 79323P    | PINK CHERRY LIGHTS                  | 12       | 2009-12-01 07:45:00 | 6.75  | 13085.0     | United Kingdom |
| 2 | 489434  | 79323W    | WHITE CHERRY LIGHTS                 | 12       | 2009-12-01 07:45:00 | 6.75  | 13085.0     | United Kingdom |
| 3 | 489434  | 22041     | RECORD FRAME 7" SINGLE SIZE         | 48       | 2009-12-01 07:45:00 | 2.10  | 13085.0     | United Kingdom |
| 4 | 489434  | 21232     | STRAWBERRY CERAMIC TRINKET BOX      | 24       | 2009-12-01 07:45:00 | 1.25  | 13085.0     | United Kingdom |

### Section 1

#### Create New Feature: Year

Membuat kolom baru dengan nama Year yang berisi nilai tahun dari Invoice Date.

```python
df['year'] = pd.to_datetime(df['InvoiceDate']).dt.year
```

#### Filtering Data

Filtering data dengan ketentuan di bawah ini dan simpan dalam variabel baru:

* Quantity minimal 1 (tidak boleh 0 dan minus)
* Kolom Invoice tidak mengandung huruf ‘C’ karena hal tersebut menandakan pelanggan tidak menyelesaikan belanjanya atau melakukan pembatalan.

```python
df_filter = df[(df['Quantity']>0) & (df['Invoice'].str.contains("C")==False)]
```

#### Create New Feature: Revenue

Membuat kolom baru bernama Revenue dengan nilai Quantity dikali dengan Price.

```python
df['Revenue'] = df.Price * df.Quantity
```

#### Average of Revenue per Year

Menghitung rata-rata Revenue per tahun.

```python
rev_mean = df.groupby('year', as_index=False)["Revenue"].mean()
rev_mean
```

|   | year | Revenue   |
| - | ---- | --------- |
| 0 | 2009 | 17.684777 |
| 1 | 2010 | 18.152555 |
| 2 | 2011 | 18.018195 |

```python
rev_mean['year'] = rev_mean['year'].astype(str)
plot1 = rev_mean.plot(x='year', rot=0, kind='line', title ="Average of Revenue per Year",figsize=(8,4),legend=True, fontsize=12)
plot1.set_xlabel("Year",fontsize=12)
plot1.set_ylabel("Averago of Revenue",fontsize=12)
plt.show()
```

![](../../.gitbook/assets/output\_15\_0.png)

#### Interpretation

* Dari grafik dapat diketahui bahwa terjadi kenaikan rata-rata pendapatan pada tahun `2010` dengan nilai 18.15 dan termasuk pendapatan rata-rata tertinggi dalam jangka waktu 3 tahun



### Section 2

#### Filtering Data

Filtering menggunakan data sales (data yang sudah difilter pada section 1) dengan ketentuan CustomerID tidak boleh kosong atau null. Kemudian simpan dalam variabel finished.

```python
df = df.rename(columns={'Customer ID': 'CustomerID'})
finished = df[df.CustomerID.notnull()]
```

**Customers who finished their purchases**

**F**iltering data untuk mengelompokkan pelanggan yang menyelesaikan belanja.

```python
finish = finished[~finished['Invoice'].str.contains("C")]
```

**Customers who canceled their purchases**

**F**iltering data untuk mengelompokkan pelanggan yang membatalkan belanjanya, dengan cara mendeteksi kolom Invoice mengandung huruf ‘C’.

```python
cancel = finished[finished['Invoice'].str.contains("C")]
```

#### Number of Finished and Canceled Transactions Each Year

Menghitung jumlah transaksi yang berhasil (dari variabel finished) dan jumlah transaksi yang dibatalkan (dari variabel cancel) untuk setiap tahunnya.

```python
count_finish = finish.groupby('year', as_index=False)['Invoice'].count()
count_finish['year'] = count_finish['year'].astype(str)
count_finish.set_index('year', inplace=True)

count_cancel = cancel.groupby('year', as_index=False)['Invoice'].count()
count_cancel['year'] = count_cancel['year'].astype(str)
count_cancel.set_index('year', inplace=True)

count_total = pd.merge(count_finish, count_cancel, left_index=True, right_index=True, suffixes=('Finish', 'Cancel'))
count_total
```

|      | InvoiceFinish | InvoiceCancel |
| ---- | ------------- | ------------- |
| year |               |               |
| 2009 | 30761         | 999           |
| 2010 | 403094        | 9530          |
| 2011 | 371765        | 8215          |

```python
plot1 = count_finish.plot(kind='line', title ="Number of Finished Transactions Each Year",figsize=(8,4),legend=True, fontsize=12)
plot1.set_xlabel("Year",fontsize=12)
plot1.set_ylabel("Number of Finished Transactions ",fontsize=12)
plt.show()
```

![](../../.gitbook/assets/output\_27\_0.png)

```python
plot1 = count_cancel.plot(kind='line', title ="Number of Canceled Transactions Each Year", figsize=(8,4), legend=True, fontsize=12, color='orange')
plot1.set_xlabel("Year",fontsize=12)
plot1.set_ylabel("Number of Canceled Transactions ",fontsize=12)
plt.show()
```

![](../../.gitbook/assets/output\_28\_0.png)

```python
plot1 = count_total.plot(kind='line', title ="Number of Finished and Canceled Transactions Each Year",figsize=(8,4),legend=True, fontsize=12)
plot1.set_xlabel("Year",fontsize=12)
plot1.set_ylabel("Number of Finished and Canceled Transactions ",fontsize=12)
plt.show()
```

![](../../.gitbook/assets/output\_29\_0.png)

#### Cancellation Rate

Menghitung cancellation rate untuk setiap tahunnya.&#x20;

> Cancellation rate adalah persentase pelanggan yang melakukan pembatalan order yang telah dilakukan. Formulanya adalah jumlah customer yang cancel dibagi jumlah seluruh customer kemudian dikali 100%.

```python
customerr = df.groupby('year', as_index=False)['CustomerID'].count()
customerr['year'] = customerr['year'].astype(str)
customerr.set_index('year', inplace=True)

cancelr = cancel.groupby('year', as_index=False)['CustomerID'].count()
cancelr['year'] = cancelr['year'].astype(str)
cancelr.set_index('year', inplace=True)

cancel_rate = ((cancelr/customerr))*100
cancel_rate
```

|      | CustomerID |
| ---- | ---------- |
| year |            |
| 2009 | 3.145466   |
| 2010 | 2.309609   |
| 2011 | 2.161956   |

```python
plot2 = cancel_rate.plot(kind='line', title ="Cancellation Rate Each Year",figsize=(8,4),legend=True, fontsize=12)
plot2.set_xlabel("Year",fontsize=12)
plot2.set_ylabel("Cancellation Rate",fontsize=12)
plt.show()
```

![](../../.gitbook/assets/output\_32\_0.png)

#### Interpretation

* Dari Grafik `Number of Finished and Canceled Transactions Each Year` dan `Cancellation Rate Each Year` di atas bisa dilihat bahwa naiknya grafik angka transaksi berasil juga selaras dengan penurunan rate cancellation pada tahun 2010.
