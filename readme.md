# Laporan Proyek Machine Learning Sistem Rekomendasi Film - Muhammad Ashim Izzuddin

## Project Overview

### Latar Belakang

Di era digital saat ini, jumlah film yang tersedia bagi pengguna sangatlah banyak. Kondisi ini menciptakan tantangan bagi pengguna untuk menemukan film yang sesuai dengan preferensi mereka, sebuah fenomena yang dikenal sebagai *information overload*. Sistem rekomendasi film hadir untuk membantu pengguna mengatasi masalah ini dengan menyarankan film-film yang kemungkinan besar akan mereka sukai, berdasarkan berbagai faktor dan data historis.

### Mengapa Proyek Ini Penting

Proyek pengembangan sistem rekomendasi film ini memiliki signifikansi yang cukup besar, baik dari sisi pengguna maupun penyedia layanan:

  - Bagi **pengguna**, sistem ini meningkatkan pengalaman menonton dengan mempermudah penemuan konten yang relevan dan menarik di antara banyaknya pilihan.
  - Bagi **penyedia layanan** (misalnya, platform streaming), sistem rekomendasi yang efektif dapat meningkatkan *engagement* (keterlibatan), kepuasan, dan retensi pengguna, yang pada akhirnya berdampak positif pada aspek bisnis.

### Referensi Terkait
  * Aggarwal, C. C. (2016). *Recommender Systems: The Textbook*. Springer.
  * Ricci, F., Rokach, L., & Shapira, B. (2011). Introduction to Recommender Systems Handbook. In *Recommender Systems Handbook* (pp. 1-35). Springer US.

---
## Business Understanding

### Problem Statements
1.  Pengguna seringkali merasa kewalahan dengan banyaknya pilihan film yang tersedia (*information overload*), sehingga kesulitan menemukan film baru yang benar-benar sesuai dengan preferensi personal mereka secara efisien.
2.  Penyedia layanan streaming film memiliki tantangan dalam mempertahankan minat pengguna dan meningkatkan durasi interaksi (*user engagement*) jika tidak dapat menyajikan konten yang relevan secara proaktif.
3.  Bagaimana cara mengembangkan sistem yang mampu memberikan rekomendasi film yang dipersonalisasi secara efektif kepada pengguna, dengan memanfaatkan histori perilaku mereka (rating) dan karakteristik intrinsik film (genre, tag)?

### Goals
1.  Mengembangkan dua model sistem rekomendasi film: satu berdasarkan kemiripan konten (Content-Based Filtering) dan satu lagi berdasarkan pola perilaku pengguna kolektif (Collaborative Filtering menggunakan SVD).
2.  Menghasilkan daftar Top-N rekomendasi film yang relevan untuk pengguna atau film tertentu berdasarkan masing-masing model.
3.  Mengevaluasi performa kedua model menggunakan metrik yang sesuai (misalnya, Presisi untuk Content-Based Filtering, dan RMSE/MAE untuk Collaborative Filtering) untuk memahami efektivitas dan karakteristik masing-masing pendekatan pada dataset yang digunakan.

### Solution Statements
1.  Solusi yang diusulkan adalah dengan mengimplementasikan **Content-Based Filtering** yang memanfaatkan fitur tekstual film (genre dan tag) yang diolah menggunakan TF-IDF dan diukur kemiripannya dengan Cosine Similarity. Pendekatan ini akan merekomendasikan film yang secara intrinsik mirip dengan film yang pernah disukai pengguna.
2.  Selain itu, akan diimplementasikan **Collaborative Filtering** menggunakan algoritma Singular Value Decomposition (SVD) untuk mengidentifikasi pola tersembunyi dalam data rating pengguna. Pendekatan ini akan merekomendasikan film berdasarkan preferensi pengguna lain yang memiliki selera serupa.
3.  Efektivitas kedua solusi akan diukur menggunakan metrik yang relevan, dan hasilnya akan dianalisis untuk menunjukkan bagaimana sistem ini menjawab permasalahan yang telah diidentifikasi.

---
## Data Understanding

### Sumber Data

  * **Dataset**: MovieLens ml-latest-small.
  * **Tautan Unduh**: [https://grouplens.org/datasets/movielens/latest-small/](https://grouplens.org/datasets/movielens/latest-small/)
  * **File yang Digunakan**: `movies.csv`, `ratings.csv`, `tags.csv`. File `links.csv` juga dimuat namun tidak menjadi fokus utama dalam pemodelan sistem rekomendasi ini.

### Informasi Data

Berikut adalah ringkasan dari masing-masing file data yang digunakan, berdasarkan analisis awal:

1.  **`movies.csv`**:
      * Jumlah Data: 9.742 film.
      * Kolom: `movieId`, `title`, `genres`.
      * Missing Values: Tidak ada nilai yang hilang.
      * Duplikasi: Tidak ada baris duplikat.
      * Deskripsi: Berisi ID unik film, judul (termasuk tahun rilis), dan genre yang dipisahkan oleh `|`. Terdapat 9.737 judul unik dan 951 kombinasi genre unik.

2.  **`ratings.csv`**:
      * Jumlah Data: 100.836 rating.
      * Kolom: `userId`, `movieId`, `rating`, `timestamp`.
      * Missing Values: Tidak ada nilai yang hilang.
      * Duplikasi: Tidak ada baris duplikat.
      * Deskripsi: Berisi rating (0.5-5.0) yang diberikan oleh 610 pengguna unik terhadap film. Rata-rata rating sekitar 3.50.

3.  **`tags.csv`**:
      * Jumlah Data: 3.683 tag.
      * Kolom: `userId`, `movieId`, `tag`, `timestamp`.
      * Missing Values: Tidak ada nilai yang hilang.
      * Duplikasi: Tidak ada baris duplikat.
      * Deskripsi: Berisi tag yang diberikan pengguna ke film. Terdapat 1.589 tag unik. Tag "In Netflix queue" adalah yang paling sering muncul.

4.  **`links.csv`**:
      * Jumlah Data: 9.742 entri.
      * Kolom: `movieId`, `imdbId`, `tmdbId`.
      * Missing Values: Kolom `tmdbId` memiliki 8 nilai yang hilang.
      * Duplikasi: Tidak ada baris duplikat.
      * Deskripsi: Berisi ID eksternal untuk menghubungkan dengan database film lain seperti IMDb dan TMDB.

### Deskripsi Variabel/Fitur

  * **`movies.csv`**:
      * `movieId`: ID unik untuk setiap film (Integer). Kunci utama.
      * `title`: Judul film, termasuk tahun rilis (String).
      * `genres`: Genre film, dipisahkan oleh `|` (String).

  * **`ratings.csv`**:
      * `userId`: ID unik untuk setiap pengguna (Integer).
      * `movieId`: ID unik untuk film yang diberi rating (Integer).
      * `rating`: Rating yang diberikan (Float, skala 0.5 - 5.0).
      * `timestamp`: Waktu pemberian rating (Integer, Unix timestamp).

  * **`tags.csv`**:
      * `userId`: ID unik pengguna yang memberikan tag (Integer).
      * `movieId`: ID unik film yang diberi tag (Integer).
      * `tag`: Tag yang diberikan (String).
      * `timestamp`: Waktu pemberian tag (Integer, Unix timestamp).

  * **`links.csv`**:
      * `movieId`: ID unik untuk setiap film (Integer).
      * `imdbId`: ID film di IMDb (Integer).
      * `tmdbId`: ID film di TMDB (Float).

### Exploratory Data Analysis (EDA) & Visualisasi

Analisis data eksploratif dilakukan untuk memahami lebih dalam karakteristik dataset.

1.  **Distribusi Rating Film**
      * **Deskripsi**: Histogram frekuensi nilai rating pengguna (0.5-5.0).
      * **Visualisasi**:
        ![Distribusi Rating Pengguna](https://github.com/user-attachments/assets/8dd84a49-9dfb-424d-a5cb-732746131a6e)

      * **Penjelasan**: Dihasilkan menggunakan `sns.histplot` pada `ratings_df['rating']` untuk menganalisis tren pemberian rating.
      * **Insight**: Rating paling sering adalah **4.0**, diikuti 3.0 dan 5.0. Pengguna cenderung memberi rating tengah ke atas. Distribusi *left-skewed*.

2.  **Distribusi Genre Film (Top 15)**
      * **Deskripsi**: Diagram batang horizontal 15 genre film paling umum.
      * **Visualisasi**:
        ![15 Genre Film Paling Umum](https://github.com/user-attachments/assets/6dac46e4-0b97-40cc-92af-0ebe0fbed950)

      * **Penjelasan**: Dibuat dengan menghitung frekuensi setiap genre dari `movies_df` dan ditampilkan dengan `sns.barplot`.
      * **Insight**: Genre dominan adalah **Drama** dan **Comedy**. Lainnya termasuk Thriller, Action, Romance. Relevan untuk Content-Based Filtering.

3.  **Word Cloud Tag Film**
      * **Deskripsi**: Visualisasi *word cloud* tag film, ukuran tag menunjukkan frekuensi.
      * **Visualisasi**:
        ![Word Cloud dari Tag Film](https://github.com/user-attachments/assets/aedbaa63-bf1f-4794-a76c-15beab51c04c)

      * **Penjelasan**: Dihasilkan dari semua tag menggunakan library `WordCloud`.
      * **Insight**: Menunjukkan tag populer seperti "sci-fi", "atmospheric", "funny", "action". Memberikan dimensi konten tambahan.

---
## Data Preparation

Proses persiapan data merupakan tahap krusial untuk memastikan data yang digunakan dalam pemodelan berkualitas baik dan sesuai dengan kebutuhan masing-masing algoritma sistem rekomendasi. Berikut adalah langkah-langkah utama yang dilakukan:

### 1. Pemuatan Data (Data Loading)
Dataset MovieLens ml-latest-small (`movies.csv`, `ratings.csv`, `tags.csv`) dimuat ke dalam DataFrame Pandas (`movies_df`, `ratings_df`, `tags_df`). Tujuannya adalah untuk mendapatkan akses ke informasi film, rating pengguna, dan tag yang akan digunakan dalam pemodelan. Hasilnya adalah DataFrame yang siap untuk diproses lebih lanjut, dengan pemeriksaan awal terhadap struktur dan isinya.

### 2. Pra-pemrosesan Data untuk Content-Based Filtering
Langkah-langkah berikut dilakukan untuk menyiapkan data yang akan digunakan oleh model Content-Based Filtering:

* **Penggabungan Data Film dengan Tag**:
    * **Deskripsi Langkah**: Informasi tag dari `tags_df` diagregasi per `movieId` (setelah diubah ke huruf kecil) dan kemudian digabungkan dengan `movies_df` untuk menghasilkan `movies_with_tags_df`.
    * **Tujuan**: Memperkaya fitur konten setiap film dengan menyertakan tag yang relevan yang diberikan oleh pengguna.
    * **Hasil**: DataFrame `movies_with_tags_df` di mana setiap film kini memiliki kolom `tag` yang berisi gabungan tag-tag yang terkait (string kosong jika tidak ada tag).

* **Pembuatan Kolom Metadata Gabungan**:
    * **Deskripsi Langkah**: Kolom `genres` pada `movies_with_tags_df` dibersihkan dengan mengganti karakter `|` dengan spasi (menjadi `genres_cleaned`). Kemudian, kolom `genres_cleaned` dan kolom `tag` digabungkan menjadi satu kolom teks baru bernama `metadata`.
    * **Tujuan**: Membuat satu representasi tekstual tunggal yang komprehensif untuk setiap film, yang akan digunakan sebagai input untuk TfidfVectorizer. Ini menyederhanakan proses ekstraksi fitur.
    * **Hasil**: DataFrame `movies_for_content_based_df` (salinan dari `movies_with_tags_df`) kini memiliki kolom `metadata` yang siap untuk di-vektorisasi.

### 3. Persiapan Data untuk Collaborative Filtering (Surprise)
Untuk model Collaborative Filtering yang menggunakan library Surprise:

* **Deskripsi Langkah**: Dataset rating (`ratings_df` yang hanya berisi kolom `userId`, `movieId`, dan `rating`) disiapkan. Objek `Reader` dari library Surprise diinisialisasi dengan skala rating yang sesuai (0.5 hingga 5.0). Kemudian, data dari DataFrame dimuat menggunakan `Dataset.load_from_df()` ke dalam format dataset internal Surprise.
* **Tujuan**: Mengubah data rating ke dalam struktur data spesifik yang dibutuhkan oleh library Surprise untuk proses pelatihan dan evaluasi model SVD.
* **Hasil**: Objek `data_surprise` yang siap digunakan untuk membangun `trainset` atau melakukan cross-validation dengan model-model dari library Surprise.

### 4. Data Akhir untuk Pemodelan
Setelah semua langkah persiapan data:

* **Untuk Content-Based Filtering**: Model akan menggunakan matriks TF-IDF yang dihasilkan dari kolom `metadata` pada `movies_for_content_based_df`, serta matriks kemiripan kosinus yang dihitung dari matriks TF-IDF tersebut.
* **Untuk Collaborative Filtering**: Model SVD akan menggunakan objek `data_surprise` yang berisi informasi rating pengguna-film.

---
## Modeling and Results

Pada tahap ini, dilakukan pengembangan dua jenis model sistem rekomendasi: Content-Based Filtering dan Collaborative Filtering menggunakan algoritma SVD. Setiap model akan dijelaskan prosesnya, justifikasi pemilihan algoritmanya, hingga contoh hasil rekomendasi yang diberikan.

### Pendekatan 1: Content-Based Filtering

Content-Based Filtering merekomendasikan item berdasarkan kemiripan fitur intrinsik item tersebut dengan item yang pernah disukai pengguna atau item yang sedang dilihat.

#### 1. Justifikasi Pemilihan Algoritma
Pendekatan Content-Based Filtering dipilih karena beberapa alasan:
* **Transparansi**: Rekomendasi yang diberikan dapat dijelaskan berdasarkan fitur-fitur spesifik film (seperti genre dan tag).
* **Mengatasi *Item Cold-Start***: Dapat memberikan rekomendasi untuk item baru selama item tersebut memiliki deskripsi fitur, tanpa memerlukan data interaksi sebelumnya.
* **Personalisasi Berdasarkan Fitur**: Mampu menangkap preferensi pengguna terhadap atribut-atribut tertentu dari sebuah film.

Implementasi teknisnya menggunakan:
* **TF-IDF (Term Frequency-Inverse Document Frequency)**: Digunakan untuk mengubah data tekstual dari fitur film (gabungan genre dan tag pada kolom `metadata`) menjadi representasi vektor numerik. TF-IDF efektif dalam memberikan bobot pada kata-kata yang penting dalam mendeskripsikan sebuah film. Parameter `stop_words='english'` digunakan untuk mengabaikan kata-kata umum dalam bahasa Inggris.
* **Cosine Similarity**: Digunakan untuk menghitung skor kemiripan antara vektor TF-IDF dari setiap pasangan film. Metrik ini mengukur kesamaan orientasi antar vektor, cocok untuk data teks yang telah divektorisasi.

#### 2. Proses Pemodelan Langkah-demi-Langkah
Proses pemodelan Content-Based Filtering melibatkan beberapa tahap utama:
1.  **Pembuatan Fitur `metadata`**: Kolom `genres` dan `tag` digabungkan menjadi satu kolom teks tunggal `metadata` untuk setiap film.
2.  **Vektorisasi dengan TF-IDF**: Kolom `metadata` ditransformasikan menjadi matriks numerik (`tfidf_matrix`) menggunakan `TfidfVectorizer`.
3.  **Perhitungan Matriks Kemiripan**: Matriks kemiripan antar film (`cosine_sim_matrix`) dihitung dengan menerapkan Cosine Similarity pada `tfidf_matrix`.
4.  **Pembuatan Fungsi Rekomendasi**: Dibuat fungsi `get_content_based_recommendations` untuk menghasilkan daftar film yang paling mirip dengan film input berdasarkan `cosine_sim_matrix`.

#### 3. Contoh Hasil Rekomendasi Content-Based
Berikut adalah contoh hasil Top-5 rekomendasi untuk film '**Toy Story (1995)**' menggunakan model Content-Based Filtering:

| No  | Title                                     | Genres                                          | Metadata (Contoh Cuplikan dari Notebook)                 | Similarity Score |
|-----|-------------------------------------------|-------------------------------------------------|----------------------------------------------------------|------------------|
| 1   | Bug's Life, A (1998)                      | Adventure\|Animation\|Children\|Comedy          | Adventure Animation Children Comedy pixar                | 0.8622           |
| 2   | Toy Story 2 (1999)                        | Adventure\|Animation\|Children\|Comedy\|Fantasy | Adventure Animation Children Comedy Fantasy an...        | 0.6440           |
| 3   | Guardians of the Galaxy 2 (2017)          | Action\|Adventure\|Sci-Fi                       | Action Adventure Sci-Fi fun                              | 0.3677           |
| 4   | Antz (1998)                               | Adventure\|Animation\|Children\|Comedy\|Fantasy | Adventure Animation Children Comedy Fantasy              | 0.3579           |
| 5   | Adventures of Rocky and Bullwinkle, The (2000) | Adventure\|Animation\|Children\|Comedy\|Fantasy | Adventure Animation Children Comedy Fantasy              | 0.3579           |

Dari tabel di atas, terlihat bahwa film-film yang direkomendasikan memiliki genre dan metadata yang relevan dengan film input 'Toy Story (1995)', menunjukkan model berhasil menangkap kemiripan berdasarkan konten.

---
### Pendekatan 2: Collaborative Filtering (SVD)

Collaborative Filtering merekomendasikan item berdasarkan pola perilaku (misalnya, rating) dari sekelompok besar pengguna.

#### 1. Justifikasi Pemilihan Algoritma
Pendekatan Collaborative Filtering dipilih karena:
* **Kemampuan Menemukan Preferensi Laten**: Dapat mengungkap preferensi tersembunyi pengguna yang tidak eksplisit dalam fitur item.
* **Serendipity**: Berpotensi menghasilkan rekomendasi yang mengejutkan namun tetap relevan.
* **Tidak Membutuhkan Fitur Item Eksplisit**: Model ini bekerja berdasarkan data interaksi (rating).

Implementasi teknisnya menggunakan:
* **SVD (Singular Value Decomposition)**: Algoritma SVD dari library `Surprise` dipilih. SVD adalah teknik faktorisasi matriks yang efektif untuk menangani data rating yang *sparse* dan menangkap faktor laten pengguna dan item. Parameter yang digunakan adalah `n_factors=100`, `n_epochs=20`, `lr_all=0.005`, `reg_all=0.02`, dan `random_state=42`.

#### 2. Proses Pemodelan Langkah-demi-Langkah
Proses pemodelan Collaborative Filtering menggunakan SVD meliputi:
1.  **Persiapan Data untuk Surprise**: Data rating (`ratings_df`) dimuat ke dalam format dataset `Surprise` menggunakan `Reader`.
2.  **Pembentukan Trainset**: Seluruh dataset rating (`data_surprise`) digunakan untuk membangun `full_trainset`.
3.  **Inisialisasi dan Pelatihan Model SVD**: Model `SVD` diinisialisasi dan dilatih (`fit`) menggunakan `full_trainset` untuk mempelajari representasi faktor laten.
4.  **Pembuatan Fungsi Rekomendasi**: Dibuat fungsi `get_collaborative_filtering_recommendations` untuk menghasilkan rekomendasi film bagi pengguna target, dengan memprediksi rating untuk film yang belum dirating dan mengurutkannya.

#### 3. Contoh Hasil Rekomendasi Collaborative Filtering (SVD)

Berikut adalah contoh hasil Top-5 rekomendasi untuk **User ID 1** menggunakan model Collaborative Filtering (SVD):

| MovieID | Title                            | Genres               | Estimated Rating |
|---------|----------------------------------|----------------------|------------------|
| 246     | Hoop Dreams (1994)               | Documentary          | 5.0000           |
| 318     | Shawshank Redemption, The (1994) | Crime\|Drama         | 5.0000           |
| 858     | Godfather, The (1972)            | Crime\|Drama         | 5.0000           |
| 912     | Casablanca (1942)                | Drama\|Romance       | 5.0000           |
| 913     | Maltese Falcon, The (1941)       | Film-Noir\|Mystery   | 5.0000           |

Rekomendasi di atas dihasilkan berdasarkan pola rating kolektif. Model SVD memprediksi bahwa User ID 1 kemungkinan akan memberikan rating tinggi untuk film-film tersebut.

---
## Evaluation

Tahap evaluasi bertujuan untuk mengukur performa dan efektivitas dari model-model sistem rekomendasi yang telah dikembangkan. Metrik yang berbeda digunakan untuk masing-masing pendekatan, disesuaikan dengan sifat dan output dari model tersebut.

### 1. Metrik Evaluasi

Metrik yang digunakan untuk mengevaluasi kedua model adalah sebagai berikut:

* **Untuk Content-Based Filtering:**
    * **Precision@k**: Metrik ini mengukur proporsi item yang relevan dari sejumlah `k` item teratas yang direkomendasikan. Formula dasarnya adalah:
        `Precision@k = (Jumlah item rekomendasi yang relevan dalam Top-k) / k`
        Dalam konteks proyek ini, sebuah film rekomendasi dianggap **'relevan'** jika ia berbagi **minimal satu genre** yang sama dengan film input yang menjadi dasar pemberian rekomendasi. Metrik ini dipilih karena model Content-Based di sini fokus pada peringkat kemiripan berdasarkan fitur, bukan prediksi rating eksplisit, sehingga metrik seperti RMSE/MAE kurang cocok.

* **Untuk Collaborative Filtering (SVD):**
    * **Root Mean Squared Error (RMSE)**: RMSE adalah akar kuadrat dari rata-rata selisih kuadrat antara rating prediksi ($\hat{y}_i$) dan rating aktual ($y_i$). RMSE memberikan bobot lebih pada kesalahan prediksi yang besar. Nilai RMSE yang lebih rendah menunjukkan performa model yang lebih baik.
        Formula: $RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$
    * **Mean Absolute Error (MAE)**: MAE adalah rata-rata dari nilai absolut selisih antara rating prediksi dan rating aktual. MAE memberikan gambaran rata-rata besarnya kesalahan prediksi tanpa memberikan bobot lebih pada kesalahan besar. Nilai MAE yang lebih rendah juga menunjukkan performa model yang lebih baik.
        Formula: $MAE = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$
    Kedua metrik ini umum digunakan untuk mengevaluasi akurasi model Collaborative Filtering yang melakukan prediksi rating.

### 2. Hasil Evaluasi Model

Berikut adalah hasil evaluasi kuantitatif untuk masing-masing model:

#### a. Content-Based Filtering
Evaluasi dilakukan dengan menghitung Precision@10 untuk film input '**Toy Story (1995)**'.
* **Precision@10**: **1.00**

#### b. Collaborative Filtering (SVD)
Model SVD dievaluasi menggunakan teknik 5-fold cross-validation pada keseluruhan dataset rating. Tabel berikut menunjukkan hasil dari setiap fold dan rata-ratanya:

| Metric        | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean   | Std    |
|---------------|--------|--------|--------|--------|--------|--------|--------|
| RMSE (testset)| 0.8684 | 0.8800 | 0.8713 | 0.8798 | 0.8727 | 0.8745 | 0.0047 |
| MAE (testset) | 0.6661 | 0.6753 | 0.6711 | 0.6757 | 0.6709 | 0.6718 | 0.0035 |
| Fit time      | 0.63   | 0.66   | 0.66   | 0.69   | 0.63   | 0.65   | 0.02   |
| Test time     | 0.06   | 0.06   | 0.13   | 0.06   | 0.06   | 0.08   | 0.03   |

Hasil rata-rata performa model SVD adalah:
* **Rata-rata RMSE**: **0.8745**
* **Rata-rata MAE**: **0.6718**

### 3. Analisis Hasil dan Diskusi

* **Content-Based Filtering**: Nilai Precision@10 sebesar **1.00** untuk film 'Toy Story (1995)' menunjukkan bahwa semua dari 10 film yang direkomendasikan berdasarkan konten dianggap relevan (karena berbagi setidaknya satu genre dengan 'Toy Story (1995)'). Ini mengindikasikan bahwa model Content-Based sangat efektif dalam mengidentifikasi dan merekomendasikan film-film yang memiliki karakteristik genre serupa dengan film input untuk contoh kasus ini. Keunggulan model ini adalah transparansinya, namun cakupan rekomendasinya mungkin terbatas pada kemiripan fitur yang eksplisit.

* **Collaborative Filtering (SVD)**: Nilai RMSE rata-rata sekitar **0.8745** dan MAE rata-rata sekitar **0.6718** (pada skala rating 0.5-5.0) menunjukkan bahwa model SVD memiliki kemampuan prediksi rating yang cukup baik. Kesalahan prediksi rata-rata yang relatif kecil mengindikasikan bahwa model mampu menangkap pola preferensi pengguna dari data historis rating secara efektif. Model ini berpotensi memberikan rekomendasi yang lebih personal dan beragam.

* **Perbandingan dan Sinergi**: Kedua model menunjukkan performa yang baik pada metriknya masing-masing. Content-Based unggul dalam memberikan rekomendasi yang dapat dijelaskan dan relevan secara fitur, sementara Collaborative Filtering (SVD) mampu menangkap preferensi laten dan memberikan prediksi rating yang akurat. Dalam sistem nyata, penggabungan kedua pendekatan (model hybrid) dapat menghasilkan sistem rekomendasi yang lebih kuat.

### 4. Keterkaitan dengan Business Understanding

Evaluasi model ini memberikan wawasan penting terkait pencapaian tujuan bisnis yang telah ditetapkan:

* **Menjawab Problem Statements:**
    1.  **Information Overload**: Kedua model membantu mengatasi masalah ini. Model Content-Based (Precision@10 = 1.00 untuk 'Toy Story (1995)') menyaring pilihan berdasarkan kemiripan fitur, sementara Collaborative Filtering (RMSE 0.8745) menyajikan film berdasarkan selera pengguna serupa, keduanya mempermudah penemuan film.
    2.  **User Engagement**: Rekomendasi yang personal dan akurat (ditunjukkan oleh metrik evaluasi) berpotensi meningkatkan keterlibatan pengguna, karena pengguna lebih sering menemukan konten yang disukai.
    3.  **Pengembangan Sistem Personalisasi Efektif**: Proyek ini berhasil mengembangkan dua pendekatan personalisasi: Content-Based berdasarkan atribut film, dan Collaborative Filtering berdasarkan perilaku kolektif.

* **Mencapai Goals Proyek:**
    1.  **Mengembangkan Dua Model Sistem Rekomendasi**: Berhasil diimplementasikan model Content-Based Filtering dan Collaborative Filtering (SVD).
    2.  **Menghasilkan Top-N Rekomendasi**: Kedua model mampu menghasilkan daftar Top-N rekomendasi, seperti dicontohkan.
    3.  **Mengevaluasi Performa Kedua Model**: Telah dilakukan evaluasi dengan metrik yang sesuai: Precision@10 untuk Content-Based (hasil: 1.00 untuk film contoh) dan RMSE/MAE untuk SVD (hasil: RMSE 0.8745, MAE 0.6718). Hasil ini memberikan dasar kuantitatif untuk menilai efektivitas.

* **Dampak Solution Statements:**
    1.  **Implementasi Content-Based Filtering (TF-IDF & Cosine Similarity)**: Solusi ini terbukti berdampak dengan menghasilkan rekomendasi yang relevan secara fitur (Precision@10 = 1.00 untuk contoh kasus).
    2.  **Implementasi Collaborative Filtering (SVD)**: Solusi ini berdampak dalam menangkap pola preferensi pengguna, ditunjukkan oleh nilai RMSE (0.8745) dan MAE (0.6718) yang relatif rendah.
    3.  **Pengukuran Efektivitas Melalui Evaluasi**: Pelaksanaan evaluasi memberikan pemahaman kuantitatif terhadap kinerja masing-masing solusi.

Secara keseluruhan, hasil evaluasi menunjukkan bahwa kedua model sistem rekomendasi yang dikembangkan memiliki performa yang menjanjikan dan berhasil menjawab permasalahan serta tujuan yang telah ditetapkan di awal proyek.

---
## Kesimpulan

Proyek ini berhasil mengembangkan dan mengevaluasi dua model sistem rekomendasi film: Content-Based Filtering dan Collaborative Filtering (SVD) menggunakan dataset MovieLens ml-latest-small.

* Model **Content-Based Filtering**, yang memanfaatkan fitur genre dan tag film dengan TF-IDF dan Cosine Similarity, menunjukkan kemampuan yang sangat baik dalam merekomendasikan film yang serupa secara fitur, dengan hasil Precision@10 sebesar 1.00 untuk film contoh 'Toy Story (1995)'.
* Model **Collaborative Filtering (SVD)** menunjukkan performa prediktif yang solid dalam memprediksi rating pengguna, dengan mencapai rata-rata RMSE sebesar 0.8745 dan MAE sebesar 0.6718 melalui 5-fold cross-validation.

Kedua pendekatan berhasil memberikan solusi untuk membantu pengguna menemukan film yang relevan di tengah banyaknya pilihan, serta menjawab tujuan bisnis yang telah ditetapkan. Pengembangan lebih lanjut dapat mengeksplorasi model hybrid untuk menggabungkan kekuatan kedua pendekatan ini.

## Saran Untuk Pengembangan Selanjutnya
- Pengembangan model Hybrid Recommender Systems yang menggabungkan kekuatan dari Content-Based dan Collaborative Filtering.
- Eksplorasi fitur tambahan seperti informasi aktor, sutradara, atau bahkan analisis sentimen dari ulasan.
- Penyesuaian parameter model lebih lanjut (hyperparameter tuning) atau penggunaan algoritma Collaborative Filtering alternatif.
- Implementasi mekanisme untuk menangani *cold-start problem* secara lebih efektif, terutama untuk pengguna baru dalam Collaborative Filtering.

## Referensi :

### Dataset & Konsep Dasar:
- Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS), 5(4), 1-19.
Tautan: https://dl.acm.org/doi/10.1145/2827872
- Ricci, F., Rokach, L., & Shapira, B. (2011). Introduction to Recommender Systems Handbook. In Recommender Systems Handbook (pp. 1-35). Springer US.
Tautan: https://link.springer.com/chapter/10.1007/978-0-387-85820-3_1
### Content-Based Filtering:
- Pazzani, M. J., & Billsus, D. (2007). Content-Based Recommendation Systems. In The Adaptive Web (pp. 325-341). Springer Berlin Heidelberg.
Tautan: https://link.springer.com/chapter/10.1007/978-3-540-72079-9_10
- Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. Information Processing & Management, 24(5), 513-523.
Tautan: https://doi.org/10.1016/0306-4573(88)90020-4

### Collaborative Filtering & SVD:
- Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix Factorization Techniques for Recommender Systems. Computer, 42(8), 30-37.
Tautan: https://ieeexplore.ieee.org/document/5197422
- Sarwar, B., Karypis, G., Konstan, J., & Riedl, J. (2001). Item-based collaborative filtering recommendation algorithms. In Proceedings of the 10th International Conference on World Wide Web (pp. 285-295).
Tautan: https://dl.acm.org/doi/10.1145/371920.372071
- Hug, N. (2020). Surprise: A Python library for recommender systems. Journal of Open Source Software, 5(52), 2surprise.
Tautan Artikel JOSS: https://joss.theoj.org/papers/10.21105/joss.02surprise
Tautan Dokumentasi Surprise: https://surprise.readthedocs.io/

### Evaluasi Sistem Rekomendasi:
- Herlocker, J. L., Konstan, J. A., Terveen, L. G., & Riedl, J. T. (2004). Evaluating collaborative filtering recommender systems. ACM Transactions on Information Systems (TOIS), 22(1), 5-53.
Tautan: https://dl.acm.org/doi/10.1145/963770.963772
- Shani, G., & Gunawardana, A. (2011). Evaluating Recommendation Systems. In Recommender Systems Handbook (pp. 257-297). Springer US.
Tautan: https://link.springer.com/chapter/10.1007/978-0-387-85820-3_8
