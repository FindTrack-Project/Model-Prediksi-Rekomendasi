import os
import random
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# --- [0] Set seed untuk hasil yang konsisten ---
SEED = 50
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --- [1] Load data ---
df = pd.read_csv("dataset/data_keuangan_bulanan.csv", sep=';')
df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%d/%m/%Y')
df['Jam'] = pd.to_datetime(df['Jam'], format='%H:%M').dt.time
df['Jumlah'] = df['Jumlah'].astype(float)

# --- [2] Filter pengeluaran dan jumlah per hari ---
df_pengeluaran = df[df['Jenis'] == 'Pengeluaran']
df_harian = df_pengeluaran.groupby('Tanggal')['Jumlah'].sum().reset_index()

# --- [3] Buat time series harian lengkap (isi nol jika tidak ada transaksi) ---
full_dates = pd.date_range(start=df_harian['Tanggal'].min(), end=df_harian['Tanggal'].max())
df_harian = df_harian.set_index('Tanggal').reindex(full_dates, fill_value=0).rename_axis('Tanggal').reset_index()

# --- [4] Scaling ---
scaler = MinMaxScaler()
scaled_amounts = scaler.fit_transform(df_harian[['Jumlah']])

# --- [5] Buat window data (30 hari input ➝ 1 output bulanan) ---
X, y = [], []
window_size = 30
step = 30  # prediksi tiap bulan

for i in range(0, len(scaled_amounts) - window_size - step + 1, step):
    X.append(scaled_amounts[i:i+window_size])
    y.append(np.sum(scaled_amounts[i+window_size:i+window_size+step]))  # jumlah bulan berikutnya

X, y = np.array(X), np.array(y)

# --- [6] Bangun dan latih model LSTM ---
model = Sequential([
    LSTM(64, activation='relu', input_shape=(window_size, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=100, verbose=0)

# --- [7] Prediksi total bulanan dan evaluasi ---
y_pred = model.predict(X)
y_pred_inv = scaler.inverse_transform(y_pred)
y_actual_inv = scaler.inverse_transform(y.reshape(-1, 1))

# --- [8] Prediksi bulan depan (berdasarkan 30 hari terakhir) ---
last_30 = df_harian['Jumlah'].values[-30:].reshape(-1, 1)
last_30_scaled = scaler.transform(last_30).reshape(1, window_size, 1)
pred_scaled_next = model.predict(last_30_scaled)
pred_rp_next = scaler.inverse_transform(pred_scaled_next)[0][0]

# --- [9] Rekomendasi alokasi pengeluaran per kategori ---
# Hitung total historis per kategori
kategori_summary = df_pengeluaran.groupby('Kategori')['Jumlah'].sum()
kategori_persen = kategori_summary / kategori_summary.sum()

# Rekomendasi alokasi berdasarkan prediksi bulan depan
rekomendasi = (kategori_persen * pred_rp_next).sort_values(ascending=False)

# --- [10] Visualisasi ---
# Ambil tanggal terakhir dari setiap window prediksi aktual
tanggal_akhir_bulan = [df_harian['Tanggal'].iloc[i+window_size+step-1] for i in range(0, len(scaled_amounts) - window_size - step + 1, step)]
bulan_labels = pd.to_datetime(tanggal_akhir_bulan).to_series().dt.to_period('M').astype(str).tolist()
next_bulan = (pd.to_datetime(bulan_labels[-1]) + pd.DateOffset(months=1)).strftime('%Y-%m')


plt.figure(figsize=(10, 5))
plt.plot(bulan_labels, y_actual_inv.flatten(), label='Aktual')
plt.plot(bulan_labels, y_pred_inv.flatten(), label='Prediksi')
plt.axvline(x=next_bulan, color='red', linestyle='--', label='Prediksi Bulan Depan')
plt.plot(next_bulan, pred_rp_next, 'ro', label='Prediksi Bulan Depan')

plt.title('Prediksi Total Pengeluaran Bulanan dan Proyeksi Bulan Depan')
plt.xlabel('Bulan')
plt.ylabel('Jumlah (Rp)')
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# Evaluasi
mae = mean_absolute_error(y_actual_inv, y_pred_inv)
mape = mean_absolute_percentage_error(y_actual_inv, y_pred_inv) * 100  # persen

print(f"MAE  (Mean Absolute Error): Rp {mae:,.0f}")
print(f"MAPE (Mean Absolute % Error): {mape:.2f}%")

# --- [11] Tampilkan hasil ---
print(f"\n✅ Prediksi total pengeluaran bulan depan: Rp {pred_rp_next:,.0f}")
print("\n📊 Rekomendasi alokasi pengeluaran per kategori:")
print(rekomendasi.round(0).astype(int))

# --- [12] Simpan model dan kategori persen ---

# Simpan model LSTM ke file .h5
model.save("model_prediksi_lstm.h5")

# Simpan scaler untuk normalisasi (penting jika ingin prediksi ulang)
import joblib
joblib.dump(scaler, "scaler.pkl")

# Simpan proporsi alokasi kategori ke JSON
import json
kategori_persen_dict = kategori_persen.to_dict()

with open("kategori_persen.json", "w") as f:
    json.dump(kategori_persen_dict, f, indent=4)

print("\n✅ Model, scaler, dan proporsi kategori berhasil disimpan.")