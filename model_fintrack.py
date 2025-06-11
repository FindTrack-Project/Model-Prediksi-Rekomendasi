import joblib
import numpy as np
from tensorflow.keras.models import load_model
import os
import boto3
from botocore.client import Config
from dotenv import load_dotenv

load_dotenv()

R2_ACCESS_KEY_ID = os.environ.get('R2_ACCESS_KEY_ID')
R2_SECRET_ACCESS_KEY = os.environ.get('R2_SECRET_ACCESS_KEY')
R2_ACCOUNT_ID = os.environ.get('R2_ACCOUNT_ID')
R2_BUCKET_NAME = os.environ.get('R2_BUCKET_NAME')

MODEL_BLOB_NAME = 'model_prediksi_lstm.h5'
SCALER_BLOB_NAME = 'scaler.pkl'

LOCAL_MODEL_PATH = '/tmp/' + MODEL_BLOB_NAME
LOCAL_SCALER_PATH = '/tmp/' + SCALER_BLOB_NAME

model = None
scaler = None

def get_r2_client():
    if not all([R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ACCOUNT_ID, R2_BUCKET_NAME]):
        print("WARNING: Cloudflare R2 credentials atau account ID/bucket name tidak lengkap di environment variables.")
        print("Pastikan R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ACCOUNT_ID, dan R2_BUCKET_NAME sudah diset.")
        return None

    r2_endpoint_url = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

    s3 = boto3.client(
        service_name='s3',
        endpoint_url=r2_endpoint_url,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(signature_version='s3v4')
    )
    return s3

def download_file_from_r2(bucket_name, blob_name, local_path):
    print(f"Mencoba mengunduh '{blob_name}' dari R2 bucket '{bucket_name}' ke '{local_path}'...")
    s3_client = get_r2_client()
    if s3_client is None:
        return False

    try:
        if os.path.exists(local_path):
            print(f"File '{blob_name}' sudah ada di {local_path}. Melewati pengunduhan.")
            return True

        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        s3_client.download_file(bucket_name, blob_name, local_path)
        print(f"File '{blob_name}' berhasil diunduh ke {local_path}")
        return True
    except Exception as e:
        print(f"Gagal mengunduh '{blob_name}' dari R2: {e}")
        return False

def load_models_and_scaler():
    global model, scaler

    if model is None:
        if R2_BUCKET_NAME and download_file_from_r2(R2_BUCKET_NAME, MODEL_BLOB_NAME, LOCAL_MODEL_PATH):
            try:
                model = load_model(LOCAL_MODEL_PATH, compile=False)
                print(f"Model '{MODEL_BLOB_NAME}' berhasil dimuat dari '{LOCAL_MODEL_PATH}'.")
            except Exception as e:
                print(f"Error memuat model dari '{LOCAL_MODEL_PATH}': {e}")
                model = None

    if scaler is None:
        if R2_BUCKET_NAME and download_file_from_r2(R2_BUCKET_NAME, SCALER_BLOB_NAME, LOCAL_SCALER_PATH):
            try:
                with open(LOCAL_SCALER_PATH, 'rb') as f:
                    scaler = joblib.load(f)
                print(f"Scaler '{SCALER_BLOB_NAME}' berhasil dimuat dari '{LOCAL_SCALER_PATH}'.")
            except Exception as e:
                print(f"Error memuat scaler dari '{LOCAL_SCALER_PATH}': {e}")
                scaler = None

load_models_and_scaler()

def predict_next_month_expense(last_n_days_data):
    if model is None or scaler is None:
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
        data_terakhir_30hari = [0] * (30 - len(last_n_days_data)) + last_n_days_data

    arr = np.array(data_terakhir_30hari).reshape(-1, 1)
    scaled_arr = scaler.transform(arr)
    reshaped = scaled_arr.reshape(1, 30, 1)

    pred_scaled = model.predict(reshaped)
    pred_rp = float(scaler.inverse_transform(pred_scaled)[0][0])

    budget_rp = float(pred_rp * 1.1)

    return pred_rp, budget_rp

if __name__ == "__main__":
    print("\n--- Pengujian model_fintrack.py dengan Cloudflare R2 ---")
    sample_data = [0, 191100, 0, 558715, 402666, 0, 751196, 169118, 0, 187833,
                   294365, 157361, 227050, 720798, 164311, 526255, 191603, 514564, 191100, 454604,
                   555158, 187833, 303180, 391541, 542234, 580565, 454604, 383170, 366019, 369494]

    try:
        load_models_and_scaler()

        if model is not None and scaler is not None:
            pred_rp, budget_rp = predict_next_month_expense(sample_data)
            print(f"💸 Prediksi Pengeluaran Bulan Depan : Rp{pred_rp:,.2f}")
            print(f"📌 Rekomendasi Budget Bulan Depan  : Rp{budget_rp:,.2f}")
        else:
            print("❌ Prediksi gagal. Model atau scaler tidak berhasil dimuat dari R2.")
    except Exception as e:
        print(f"Terjadi error saat pengujian: {e}")