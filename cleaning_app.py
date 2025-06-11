import streamlit as st
import pandas as pd

# Atur konfigurasi halaman
st.set_page_config(page_title="Aplikasi Pembersih Data", layout="centered")

# Judul aplikasi
st.title("Aplikasi Pembersih Data Sederhana")

# Upload file Excel
fileDiupload = st.file_uploader("Unggah file Excel", type=["xlsx"])

if fileDiupload:
    # Baca file Excel
    data = pd.read_excel(fileDiupload, header=0)

    # Ganti nama kolom
    data.columns = [
        "Harga",
        "LuasTanah",
        "LuasBangunan",
        "JumlahKamarTidur",
        "JumlahKamarMandi",
        "Garasi",
        "Kota"
    ]

    # Hapus baris pertama (biasanya judul ganda)
    data = data.drop(index=0)

    # Tampilkan data asli
    st.subheader("Data Asli")
    st.dataframe(data.head())

    # Fungsi untuk menghapus nilai kosong
    def hapusKosong(data, daftarKolom):
        for kolom in daftarKolom:
            data = data[data[kolom].notnull()]
        return data

    # Fungsi untuk konversi tipe data
    def konversiTipe(data, tipeData):
        for kolom, tipe in tipeData.items():
            if kolom in data.columns:
                data[kolom] = data[kolom].astype(tipe)
        return data

    # Daftar kolom yang perlu dicek
    daftarKolom = [
        "Harga",
        "LuasTanah",
        "LuasBangunan",
        "JumlahKamarTidur",
        "JumlahKamarMandi",
        "Garasi",
        "Kota"
    ]

    # Hapus baris dengan nilai kosong
    data = hapusKosong(data, daftarKolom)

    # Tentukan tipe data yang diinginkan
    tipeData = {
        "Harga": float,
        "LuasTanah": int,
        "LuasBangunan": int,
        "JumlahKamarTidur": int,
        "JumlahKamarMandi": int,
        "Garasi": str,
        "Kota": str
    }

    # Konversi tipe data
    data = konversiTipe(data, tipeData)

    # Tampilkan data setelah dibersihkan
    st.subheader("Data Setelah Dibersihkan")
    st.dataframe(data.head())

    # Simpan sebagai file CSV
    hasilCSV = data.to_csv(index=False).encode("utf-8")

    # Tombol unduh
    st.download_button(
        label="Unduh Data CSV",
        data=hasilCSV,
        file_name="HARGA RUMAH JAKSEL_clean.csv",
        mime="text/csv"
    )
