# model_fintrack.py
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import os

# --- Inisialisasi Model dan Scaler ---

MODEL_PATH = 'model_prediksi_lstm.h5'
SCALER_PATH = 'scaler.pkl'

model = None
scaler = None

# Load model LSTM
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH, compile=False)
        print(f"Model '{MODEL_PATH}' berhasil dimuat.")
    except Exception as e:
        print(f"Error memuat model: {e}")
else:
    print(f"File model '{MODEL_PATH}' tidak ditemukan.")

# Load scaler dari file pickle
if os.path.exists(SCALER_PATH):
    try:
        with open(SCALER_PATH, 'rb') as f:
            scaler = joblib.load(f)
        print("Scaler berhasil dimuat dari 'scaler.pkl'.")
    except Exception as e:
        print(f"Error memuat scaler: {e}")
else:
    print(f"File scaler '{SCALER_PATH}' tidak ditemukan.")

# --- Fungsi Prediksi ---

def predict_next_month_expense(last_n_days_data):
    """
    Memprediksi pengeluaran bulan depan dari pengeluaran harian terakhir.

    Parameters:
        last_n_days_data (list of float): Data pengeluaran harian terakhir.
                                           Bisa <30, =30, atau >30 data.

    Returns:
        tuple:
            - prediksi_pengeluaran (float)
            - budget_rp (float)
    """

    if model is None:
        raise RuntimeError("Model belum dimuat. Pastikan file model ada di path yang benar dan formatnya valid.")
    if scaler is None:
        raise RuntimeError("Scaler belum dimuat. Pastikan file scaler (.pkl) ada dan bisa dibaca.")

    # Validasi input
    if not isinstance(last_n_days_data, (list, np.ndarray)):
        raise ValueError("Input harus berupa list atau array.")
    if not all(isinstance(x, (int, float)) for x in last_n_days_data):
        raise ValueError("Semua elemen harus berupa angka.")
    if any(x < 0 for x in last_n_days_data):
        raise ValueError("Input tidak boleh mengandung nilai negatif.")

    # Ambil 30 hari terakhir, atau pad dengan 0 jika kurang dari 30
    if len(last_n_days_data) >= 30:
        data_terakhir_30hari = last_n_days_data[-30:]
    else:
        data_terakhir_30hari = [0] * (30 - len(last_n_days_data)) + last_n_days_data

    # Scaling dan reshape
    arr = np.array(data_terakhir_30hari).reshape(-1, 1)
    scaled_arr = scaler.transform(arr)
    reshaped = scaled_arr.reshape(1, 30, 1)

    # Prediksi dan inverse transform
    pred_scaled = model.predict(reshaped)
    pred_rp = scaler.inverse_transform(pred_scaled)[0][0]

    # Rekomendasi budget (tambah 10%)
    budget_rp = pred_rp * 1.1

    return pred_rp, budget_rp

# --- Tes lokal ---
if __name__ == "__main__":
    print("\n--- Pengujian model_fintrack.py ---")
    # sample_data = [25000, 30000, 28000] * 11  
    sample_data = [0, 191100, 0, 558715, 402666, 0, 751196, 169118, 0, 187833,  
                    294365, 157361, 227050, 720798, 164311, 526255, 191603, 514564, 191100, 454604,  
                    555158, 187833, 303180, 391541, 542234, 580565, 454604, 383170, 366019, 369494]
    pred_rp, budget_rp = predict_next_month_expense(sample_data)

    if pred_rp is not None and budget_rp is not None:
        print(f"💸 Prediksi Pengeluaran Bulan Depan : Rp{pred_rp:,.2f}")
        print(f"📌 Rekomendasi Budget Bulan Depan  : Rp{budget_rp:,.2f}")
    else:
        print("❌ Prediksi gagal. Pastikan input valid dan model berhasil diproses.")
