---
description: Rangkuman dari buku Practical Statistics for Data Scientists
---

# Estimasi Lokasi

Variable dengan data terukur/hitungan bisa memiliki ratusan nilai yang berbeda.

_**Mean**_ -> penjumlahan semua nilai dibagi dengan banyak nilai. (average)

$$
\bar{x}=\frac{\sum^n_ix_i}{n}
$$

* $$\bar{x}$$ Mewakili rata-rata sampel dari populasi.
* n/N adalah jumlah data.
* Dalam statistika, kapital menandakan populasi dan non kapial menandakan sampel dari populasi.

_**Weighted mean**_ -> penjumlahan semua nilai dikali berat dikali jumlah berat. **Rata-rata dengan bobot yang berbeda.**

$$
\bar{x}_w=\frac{\sum^n_{i=1}w_ix_i}{\sum^n_iw_i}
$$

​Beberapa nilai lebih bervarisasi dari yang lain dan variabel observasi tinggi biasanya memiliki bobot yang lebih rendah. Contohnya, ketika kita mengambil rata-rata dari banyak sensor dan terdapat satu sensor yang kurang akurat, maka kita mungkin untuk menurunkan bobot data dari sensor itu.

* Data yang dikumpulkan tidak mewakili setiap kelompok yang ingin kita ukur.

_**Median**_ -> nilai yang membagi persis setengah dari data. Nilai yang setengah datanya ada di atas dan di bawah nilai tersebut.

* Bukan nilai tengah sebenarnya dari sebuah dataset, tapi nilai rata-rata dari dua data yang membagi data urut menjadi separuh di atas dan separuh di bawah nilai tersebut.
* Dibanding rata-rata yang menggunakaan semua pengamatan, median bergantung hanya pada nilai tengah dari data urut.
* Contoh: mencari tahu pendapatan rumah tangga masyarakat sekitar Danau Washington di Seattle. Kalau dibandingkan, lingkungan Medina dan lingkungan Windermere, jika diukur menggunakan rata-rata maka akan menghasilkan nilai yang jauh berbeda karena Bill gates tinggal di Medina. Kalau menggunakan median, seberapa kaya apa pun Bill Gates tidak akan mempengaruhi nilai tengah.

_**Weighted median**_ -> nilai yang setengah data jumlah beratnya di atas dan di bawah data urut.

* Nilai jumlah bobot yang sama dengan setengah atas dan setengah bawah dari daftar urut.
* Bersifat kokoh dari outliers.

_**Trimmed mean**_ -> rata-rata dari semua nilai setelah menurunkan nilai tetap dari nilai ekstrim (nilai terbesar dan terkecil). **Rata-rata dengan menghilangkan nilai terbesar dan terkecil dari data urut.**

$$
\bar{x}=\frac{\sum^{n-p}_{i=p+1}x_{(i)}}{n-2p}
$$

* Menyingkirkan pengaruh nilai ekstrim. Contoh, menyingkirkan skor terbesar dan terkecil dari 5 juri dalam lomba renang internasional, maka skor akhir yang dihitung adalah skor 3 juri lain. Hal ini akan mempersulit seorang juri untuk memanipulasi skor.

_**Robust**_ -> tidak sensitif terhadap nilai ekstrim. (resistant)

* Bukan cuma median yang tahan terhadap outliers, _trimmed mean_ juga bisa.
* Trimmed mean dapat dianggap sebagai kompromi antara rata-rata dan median. Hal ini karena kokoh terhadap nilai ekstrim, namun membutuhkan data lebih banyak untuk menghitung perkiraan lokasi.

_**Outlier**_ -> nilai data yang sangat berbeda dari kebanyakan data. (nilai ekstrim)

* Median tidak mudah terpengaruh oleh outliers yang dapat menyimpangkan hasil.
* Keberadaannya sering dipakai di ringkasan data dan plot data.
* Contoh outliers: beda satuan, sensor baca yang buruk.

Mean tidak selalu terbaik digunakan untuk mengukur nilai tengah.

Dalam _Anomaly detection_, outliers menjadi perhatian. Data yang besar biasanya menentukan kondisi "normal" untuk menentukan anomali.
