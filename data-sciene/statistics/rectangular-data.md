---
description: Practical Statistics for Data Scientists
---

# Rectangular Data

Berbentuk _spreadsheet_ atau tabel _database_.

* _Data frame_ -> data struktur dasar dalam model statistika dan _machine learning_.
* _Feature_ -> **kolom** dalam sebuah tabel. (variabel, input, atribut)
* _Outcome_ -> **hasil** yang diprediksi. _Feature_ biasa digunakan untuk memprediksi sebuah _outcome_ dari eksperimen. (variabel dependen, respon, output, target)
* _Records_ -> **baris** dalam sebuah tabel. (instance, pattern, sample, example, kasus)

Data _rectangular_ umumnya adalah **matriks dua dimensi** dengan baris sebagai records/kasus dan kolom sebagai features/variabel.

Data tidak selalu langsung teratur, data yang tidak terstruktur harus diproses dan dimanipulasi agar bisa berbentuk rectangular data.

Data yang saling berhubungan harus diekstrak dan dijadikan satu tabel.

Perbedaan terminologi -> statistisi menggunakan _**predictor variable**_ untuk memprediksi _**response/dependent variable**_. Ilmuwan data menggunakan _**features**_ untuk memprediksi _**target**_. Bagi ilmuwan komputer, **sampel** berarti satu baris. Sementara bagi statistisi, **sampel** berarti kumpulan beberapa baris.

_**Nonrectangular data**_

* _**Time series**_ -> perekaman data berturut-turut dengan variabel yang sama. Material mentah untuk metode peramalan statistika, juga komponen kunci data yang dihasilkan dari perangkat.
* _**Spatial data**_ -> digunakan untuk memetakan dan analisis **lokasi**, lebih kompleks dan bervariasi dari struktur data persegi. _The Object representation_, berfokus pada objek (misalnya rumah) dan koordinat tempat. _The field view_, fokus pada unit kecil dari ruang dan nilai metrik yang relevan (misalnya kecerahan pixel).
* _**Graph/network data**_ -> digunakan untuk mengambarkan hubungan fisik, sosial, dan abstrak. Contoh, grafik jejaring pertemanan (di Facebook atau LinkedIn) yang memiliki koneksi antara satu dengan yang lain dalam sebuah jejaring; Jalur distribusi yang menghubungkan jalan-jalan adalah contoh jejaring fisik. Berfungsi untuk optimalisasi jejaring dan sistem rekomendasi.
