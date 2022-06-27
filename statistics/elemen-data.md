---
description: Rangkuman dari buku Practical Statistics for Data Scientists
---

# Elemen Data

**Sumber data**

* Pengukuran sensor
* Kejadian
* Teks
* Gambar
* Video

Kebanyakan data tidak terstruktur. Gambar terdiri dari banyak pixel yang setiap pixelnya berwarna RGB, teks dapat berbentuk huruf, angka, karakter, dsb. Tantangan sains data adalah **mengubah data mentah menjadi informasi yang dapat dimengerti**.

**Tipe data**

1. Continuous -> bernilai interval. (interval, float, numeric)
2. Diskrit -> nilai bilangan asli. (integer, count)
3. Kategori -> nilai spesifik yang mewakilkan setiap kategori yang mungkin. (Enumerated, factors, nominal, Polychotomous)
4. Biner -> data kategori **spesial** yang bernilai 1/0, atau benar/salah. (dikotomi, logika, indikator, boolean)
5. Ordinal -> data kategori yang mempunyai urutan. (ordered factor)

**Struktur data**

* Numerik
* Continuous -> kecepatan angin, durasi waktu.
* Diskrit -> jumlah kejadian suatu peristiwa.
* Kategorik
* Nilai tetap -> tipe layar TV, nama negara.
* Biner -> ya/tidak, benar/salah, 0/1
* Ordinal -> rating angka (1, 2, 3, 4, 5)

Kenapa penting untuk menggolongkan tipe data? Untuk membantu menentukan tipe tampilan visual, data analisis, dan model statistika.

Keuntungan identifikasi data:

1. Mengetahui sebuah data adalah kategorik dapat memudahkan software dalam proses prosedur statistika, seperti menghasilkan bagan atau model yang tepat. Data ordinal juga bisa direpresentasikan sebagai sebuah urutan.
2. Mengoptimalisasi penyimpanan dan index.
3. Mengambil nilai mungkin dari variabel kategorik.
