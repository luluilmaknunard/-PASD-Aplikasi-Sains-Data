import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBRegressor

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Prediksi Harga Rumah", layout="centered")
st.title("Prediksi Harga Rumah di Jakarta Selatan")

# Upload file (CSV/XLSX)
fileDiupload = st.file_uploader(
    "Upload file data harga rumah (.csv)",
    type=["csv"]
)

# Jika file berhasil diupload
if fileDiupload is not None:
    # Membaca file CSV 
    if fileDiupload.name.endswith(".csv"):
        dataFrame = pd.read_csv(fileDiupload)
    else:
        st.error("Format file tidak dikenali. Harap unggah .csv atau .xlsx.")
        st.stop()

    st.success("File berhasil dibaca")

    # Kelas untuk preprocessing data
    class PraprosesHargaRumah:
        def __init__(self, dataFrame):
            self.data = dataFrame
            self.x = None
            self.y = None
            self.xTerenkode = None

        def siapkanData(self):
            fitur = ["LuasTanah", "LuasBangunan", "JumlahKamarTidur", "JumlahKamarMandi"]
            self.x = self.data[fitur]
            self.y = self.data["Harga"]

        def enkodeKategorikal(self):
            self.xTerenkode = self.x.copy()  # Tidak ada fitur kategorikal

        def praproses(self):
            self.siapkanData()
            self.enkodeKategorikal()
            return self.xTerenkode, self.y

        def bagiData(self, testSize=0.2, seed=42):
            return train_test_split(self.xTerenkode, self.y, test_size=testSize, random_state=seed)

    # Kelas untuk melatih model Random Forest
    class PelatihModelRF:
        def __init__(self, direktoriCache="cache"):
            self.direktoriCache = direktoriCache
            self.pathModelRF = os.path.join(direktoriCache, "rf_model_cache.joblib")
            os.makedirs(self.direktoriCache, exist_ok=True)

        def modelTersedia(self):
            return os.path.exists(self.pathModelRF)

        def muatModel(self):
            return joblib.load(self.pathModelRF)

        def latihModel(self, xTrain, yTrain):
            if self.modelTersedia():
                st.info("Model Random Forest dimuat dari cache")
                return self.muatModel()

            st.info("Melatih model Random Forest...")
            parameterGrid = {
                "n_estimators": [50, 100],
                "max_depth": [5, 10],
                "min_samples_split": [5],
                "min_samples_leaf": [2],
            }
            gridSearch = GridSearchCV(RandomForestRegressor(random_state=42), parameterGrid, cv=10)
            gridSearch.fit(xTrain, yTrain)
            modelTerbaik = gridSearch.best_estimator_
            st.write("Parameter terbaik Random Forest:", gridSearch.best_params_)

            joblib.dump(modelTerbaik, self.pathModelRF)
            return modelTerbaik

    # Kelas untuk melatih model XGBoost
    class PelatihModelXGB:
        def __init__(self, pathModel):
            self.pathModel = pathModel

        def modelTersedia(self):
            return os.path.exists(self.pathModel)

        def muatModel(self):
            return joblib.load(self.pathModel)

        def latihModel(self, xTrain, yTrain):
            if self.modelTersedia():
                st.info("Model XGBoost dimuat dari cache")
                return self.muatModel()

            st.info("Melatih model XGBoost...")
            parameterGrid = {
                "n_estimators": [100, 200],
                "max_depth": [3, 6],
                "learning_rate": [0.1, 0.01],
                "gamma": [0, 0.1],
            }
            gridSearch = GridSearchCV(XGBRegressor(random_state=42), parameterGrid, cv=10)
            gridSearch.fit(xTrain, yTrain)
            modelTerbaik = gridSearch.best_estimator_
            st.write("Parameter terbaik XGBoost:", gridSearch.best_params_)

            joblib.dump(modelTerbaik, self.pathModel)
            return modelTerbaik

    # Fungsi untuk melatih meta model
    def latihMetaModel(modelRF, modelXGB, xTrain, yTrain):
        predRF = modelRF.predict(xTrain)
        predXGB = modelXGB.predict(xTrain)

        dataGabungan = pd.DataFrame({
            "PrediksiRF": predRF,
            "PrediksiXGB": predXGB
        })

        metaModel = RandomForestRegressor(random_state=42)
        metaModel.fit(dataGabungan, yTrain)
        return metaModel

    # Praproses data
    praprosesor = PraprosesHargaRumah(dataFrame)
    xTerenkode, y = praprosesor.praproses()
    xTerenkode = xTerenkode.fillna(0)
    fiturFinal = xTerenkode.columns.to_list()

    xTrain, xTest, yTrain, yTest = praprosesor.bagiData()
    xTrain = xTrain[fiturFinal].fillna(0)
    xTest = xTest[fiturFinal].fillna(0)

    # Latih atau muat model Random Forest
    pelatihRF = PelatihModelRF()
    modelRF = pelatihRF.latihModel(xTrain, yTrain)

    # Latih atau muat model XGBoost
    pelatihXGB = PelatihModelXGB("cache/xgb_model_cache.pkl")
    modelXGB = pelatihXGB.latihModel(xTrain, yTrain)

    # Latih atau muat meta model
    pathMetaModel = "cache/meta_model.joblib"
    if os.path.exists(pathMetaModel):
        st.info("Meta model dimuat dari cache")
        metaModel = joblib.load(pathMetaModel)
    else:
        metaModel = latihMetaModel(modelRF, modelXGB, xTrain, yTrain)
        joblib.dump(metaModel, pathMetaModel)
        st.success("Meta model berhasil dilatih dan disimpan")

    st.success("Model siap digunakan")

    # Tampilkan fitur yang digunakan model
    st.write("Fitur yang digunakan oleh model:", fiturFinal)

    # Input prediksi dari sidebar
    st.sidebar.header("Masukkan Fitur Rumah")

    luasTanah = st.sidebar.number_input("Luas Tanah (m²)", min_value=10, max_value=10000, value=1248)
    luasBangunan = st.sidebar.number_input("Luas Bangunan (m²)", min_value=10, max_value=10000, value=1000)
    jumlahKamarTidur = st.sidebar.number_input("Jumlah Kamar Tidur", min_value=1, max_value=20, value=8)
    jumlahKamarMandi = st.sidebar.number_input("Jumlah Kamar Mandi", min_value=1, max_value=20, value=9)

    inputPengguna = {
        "LuasTanah": luasTanah,
        "LuasBangunan": luasBangunan,
        "JumlahKamarTidur": jumlahKamarTidur,
        "JumlahKamarMandi": jumlahKamarMandi
    }

    dataInput = pd.DataFrame([inputPengguna])
    dataInput = dataInput.reindex(columns=fiturFinal, fill_value=0)

    # Tombol untuk prediksi harga rumah
    if st.sidebar.button("Prediksi Harga"):
        try:
            prediksiRF = modelRF.predict(dataInput)[0]
            prediksiXGB = modelXGB.predict(dataInput)[0]

            inputMeta = pd.DataFrame({
                "PrediksiRF": [prediksiRF],
                "PrediksiXGB": [prediksiXGB]
            })

            hasilPrediksi = metaModel.predict(inputMeta)[0]
            st.subheader("Hasil Prediksi Harga Rumah:")
            st.success(f"Rp {hasilPrediksi:,.0f}".replace(",", "."))
        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
else:
    st.info("Silakan upload file .csv atau .xlsx terlebih dahulu")
