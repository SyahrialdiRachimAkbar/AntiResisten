## Ruang Lingkup Proyek: Prediksi Resistensi Antibiotik MRSA

Proyek ini bertujuan untuk mengembangkan model machine learning yang dapat memprediksi resistensi Methicillin-resistant Staphylococcus aureus (MRSA) terhadap antibiotik tertentu.

**1. Target Bakteri dan Antibiotik:**
   - **Bakteri Fokus:** Methicillin-resistant Staphylococcus aureus (MRSA).
   - **Antibiotik (Contoh):** Vancomycin, Linezolid, Daptomycin, Ceftaroline, Trimethoprim-sulfamethoxazole. (Daftar antibiotik akan disesuaikan berdasarkan ketersediaan data pada dataset yang ditemukan).

**2. Fitur Data untuk Prediksi (Ideal, akan disesuaikan berdasarkan dataset):**
   - **Data Genomik Bakteri (jika tersedia dan layak):** Kehadiran gen resistensi spesifik (misalnya, *mecA*), profil sekuens genomik.
   - **Data Pasien (jika dataset mencakup ini):**
     - Usia
     - Jenis kelamin
     - Riwayat rawat inap sebelumnya
     - Riwayat penggunaan antibiotik sebelumnya
     - Kondisi komorbiditas (misalnya, diabetes, imunosupresi)
   - **Data Klinis (jika dataset mencakup ini):**
     - Sumber/lokasi infeksi (misalnya, kulit, darah, paru-paru)
     - Hasil uji kerentanan antibiotik (AST) laboratorium (sebagai target variabel).
   - **Karakteristik Antibiotik (jika relevan dan data tersedia):**
     - Kelas antibiotik
     - Mekanisme aksi

**3. Output Model:**
   - Prediksi biner: "Resisten" atau "Rentan" terhadap antibiotik tertentu untuk strain MRSA yang diuji.
   - Kemungkinan probabilitas resistensi (jika model mendukung).

**4. Metrik Evaluasi Model:**
   - Akurasi (Accuracy)
   - Presisi (Precision)
   - Recall (Sensitivity)
   - F1-score
   - Area Under the ROC Curve (AUC-ROC)

**5. Fungsionalitas Aplikasi Streamlit:**
   - **Input Pengguna:** Memungkinkan pengguna memasukkan fitur-fitur yang relevan (misalnya, data pasien jika digunakan, atau fitur bakteri jika fokusnya hanya pada strain).
   - **Pemilihan Antibiotik:** Memungkinkan pengguna memilih antibiotik dari daftar yang didukung model untuk diprediksi resistensinya.
   - **Output Prediksi:** Menampilkan hasil prediksi (Resisten/Rentan) dengan jelas.
   - **Informasi Tambahan:** Memberikan penjelasan singkat tentang MRSA, antibiotik yang dipilih, dan interpretasi hasil prediksi.
   - **Visualisasi (Opsional):** Grafik sederhana yang menunjukkan kontribusi fitur (jika model mendukung interpretasi seperti itu, misal SHAP values untuk model tree-based).

**Catatan:** Ruang lingkup ini, terutama terkait fitur data, akan sangat bergantung pada dataset yang berhasil ditemukan dan dikumpulkan pada langkah berikutnya. Fleksibilitas akan diperlukan untuk menyesuaikan fitur berdasarkan data yang tersedia.
