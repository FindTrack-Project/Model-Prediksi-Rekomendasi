import joblib
import numpy as np
from tensorflow.keras.models import load_model
import os
import boto3 # Untuk Cloudflare R2 (kompatibel dengan S3)
from botocore.client import Config # Untuk konfigurasi endpoint S3
from dotenv import load_dotenv # <-- Tambahkan ini

# Muat variabel lingkungan dari file .env
load_dotenv() # <-- Tambahkan ini. Ini harus dipanggil SEBELUM Anda mengakses os.environ.get()

# --- Konfigurasi Cloudflare R2 (Ambil dari Variabel Lingkungan) ---
# Pastikan variabel lingkungan ini diset di Vercel Dashboard Anda DAN di file .env untuk lokal
R2_ACCESS_KEY_ID = os.environ.get('R2_ACCESS_KEY_ID')
R2_SECRET_ACCESS_KEY = os.environ.get('R2_SECRET_ACCESS_KEY')
R2_ACCOUNT_ID = os.environ.get('R2_ACCOUNT_ID')
R2_BUCKET_NAME = os.environ.get('R2_BUCKET_NAME') # Contoh: 'my-ml-models-r2-bucket'

# Nama file model di R2 bucket Anda
MODEL_BLOB_NAME = 'model_prediksi_lstm.h5'
SCALER_BLOB_NAME = 'scaler.pkl'

# Path lokal untuk menyimpan model dan scaler yang diunduh.
# Di lingkungan serverless seperti Vercel, /tmp/ adalah satu-satunya direktori writable.
LOCAL_MODEL_PATH = '/tmp/' + MODEL_BLOB_NAME
LOCAL_SCALER_PATH = '/tmp/' + SCALER_BLOB_NAME

# Variabel global untuk model dan scaler
model = None
scaler = None

# --- Fungsi untuk Menginisialisasi Klien R2 ---
def get_r2_client():
    """Menginisialisasi klien boto3 untuk Cloudflare R2."""
    if not all([R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ACCOUNT_ID, R2_BUCKET_NAME]): # Tambahkan R2_BUCKET_NAME juga
        print("WARNING: Cloudflare R2 credentials atau account ID/bucket name tidak lengkap di environment variables.")
        print("Pastikan R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ACCOUNT_ID, dan R2_BUCKET_NAME sudah diset.")
        return None # Atau raise ValueError, tergantung kebutuhan error handling Anda

    # Bentuk endpoint R2
    r2_endpoint_url = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

    s3 = boto3.client(
        service_name='s3',
        endpoint_url=r2_endpoint_url,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(signature_version='s3v4') # Penting untuk R2
    )
    return s3

# --- Fungsi untuk Mengunduh File dari R2 ---
def download_file_from_r2(bucket_name, blob_name, local_path):
    """Mengunduh file dari Cloudflare R2."""
    print(f"Mencoba mengunduh '{blob_name}' dari R2 bucket '{bucket_name}' ke '{local_path}'...")
    s3_client = get_r2_client()
    if s3_client is None:
        return False

    try:
        # Cek apakah file sudah ada di /tmp/
        if os.path.exists(local_path):
            print(f"File '{blob_name}' sudah ada di {local_path}. Melewati pengunduhan.")
            return True

        # Pastikan direktori /tmp/ ada (seharusnya sudah ada di lingkungan Vercel)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        s3_client.download_file(bucket_name, blob_name, local_path)
        print(f"File '{blob_name}' berhasil diunduh ke {local_path}")
        return True
    except Exception as e:
        print(f"Gagal mengunduh '{blob_name}' dari R2: {e}")
        return False

# --- Fungsi untuk Memuat Model dan Scaler ---
def load_models_and_scaler():
    """Mengunduh dan memuat model dan scaler jika belum dimuat."""
    global model, scaler

    # Hanya unduh dan muat jika belum ada di memori
    if model is None:
        # Pastikan R2_BUCKET_NAME tidak None sebelum mencoba mengunduh
        if R2_BUCKET_NAME and download_file_from_r2(R2_BUCKET_NAME, MODEL_BLOB_NAME, LOCAL_MODEL_PATH):
            try:
                model = load_model(LOCAL_MODEL_PATH, compile=False)
                print(f"Model '{MODEL_BLOB_NAME}' berhasil dimuat dari '{LOCAL_MODEL_PATH}'.")
            except Exception as e:
                print(f"Error memuat model dari '{LOCAL_MODEL_PATH}': {e}")
                model = None # Set ke None jika gagal

    if scaler is None:
        # Pastikan R2_BUCKET_NAME tidak None sebelum mencoba mengunduh
        if R2_BUCKET_NAME and download_file_from_r2(R2_BUCKET_NAME, SCALER_BLOB_NAME, LOCAL_SCALER_PATH):
            try:
                with open(LOCAL_SCALER_PATH, 'rb') as f:
                    scaler = joblib.load(f)
                print(f"Scaler '{SCALER_BLOB_NAME}' berhasil dimuat dari '{LOCAL_SCALER_PATH}'.")
            except Exception as e:
                print(f"Error memuat scaler dari '{LOCAL_SCALER_PATH}': {e}")
                scaler = None # Set ke None jika gagal

# --- Panggil fungsi pemuatan saat modul diimpor (atau aplikasi Flask dimulai) ---
# Ini akan memastikan model dan scaler tersedia saat cold start
load_models_and_scaler()

# --- Fungsi Prediksi (Sama seperti sebelumnya) ---
def predict_next_month_expense(last_n_days_data):
    # Pastikan model dan scaler sudah dimuat
    if model is None or scaler is None:
        # Coba muat ulang jika ada masalah saat startup (misalnya, di hot start pertama)
        load_models_and_scaler()
        if model is None or scaler is None:
            raise RuntimeError("Model atau Scaler belum dimuat. Pastikan file ada di R2 dan kredensial valid.")

    if not isinstance(last_n_days_data, (list, np.ndarray)):
        raise ValueError("Input harus berupa list atau array.")
    if not all(isinstance(x, (int, float)) for x in last_n_days_data):
        raise ValueError("Semua elemen harus berupa angka.")
    if any(x < 0 for x in last_n_days_data):
        raise ValueError("Input tidak boleh mengandung nilai negatif.")

    if len(last_n_days_data) >= 30:
        data_terakhir_30hari = last_n_days_data[-30:]
    else:
        # Isi dengan nol di awal jika data kurang dari 30 hari
        data_terakhir_30hari = [0] * (30 - len(last_n_days_data)) + last_n_days_data

    arr = np.array(data_terakhir_30hari).reshape(-1, 1)
    scaled_arr = scaler.transform(arr)
    reshaped = scaled_arr.reshape(1, 30, 1)

    pred_scaled = model.predict(reshaped)
    pred_rp = float(scaler.inverse_transform(pred_scaled)[0][0])

    budget_rp = float(pred_rp * 1.1)

    return pred_rp, budget_rp

# --- Bagian Pengujian Lokal (Hanya berjalan jika file ini dieksekusi langsung) ---
if __name__ == "__main__":
    print("\n--- Pengujian model_fintrack.py dengan Cloudflare R2 ---")
    sample_data = [0, 191100, 0, 558715, 402666, 0, 751196, 169118, 0, 187833,
                    294365, 157361, 227050, 720798, 164311, 526255, 191603, 514564, 191100, 454604,
                    555158, 187833, 303180, 391541, 542234, 580565, 454604, 383170, 366019, 369494]

    # Pastikan variabel lingkungan diset di file .env Anda di root proyek
    # dan Anda telah menginstal python-dotenv.

    try:
        # Panggil ulang pemuatan untuk memastikan semua sudah diatur jika diuji langsung
        # Ini penting jika model/scaler awalnya gagal dimuat karena belum ada .env
        load_models_and_scaler()

        if model is not None and scaler is not None:
            pred_rp, budget_rp = predict_next_month_expense(sample_data)
            print(f"💸 Prediksi Pengeluaran Bulan Depan : Rp{pred_rp:,.2f}")
            print(f"📌 Rekomendasi Budget Bulan Depan  : Rp{budget_rp:,.2f}")
        else:
            print("❌ Prediksi gagal. Model atau scaler tidak berhasil dimuat dari R2.")
    except Exception as e:
        print(f"Terjadi error saat pengujian: {e}")