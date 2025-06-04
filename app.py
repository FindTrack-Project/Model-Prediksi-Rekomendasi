from flask import Flask, request, jsonify
from flask_cors import CORS
from model_fintrack import predict_next_month_expense

app = Flask(__name__)
CORS(app)  # Mengaktifkan CORS secara global

@app.route('/predict_expense', methods=['POST'])
def predict_expense():
    try:
        data = request.get_json()
        if not data or 'last_n_days_data' not in data:
            return jsonify({'error': 'Invalid request body. Please provide "last_n_days_data".'}), 400

        last_n_days_data = data['last_n_days_data']

        # Validasi format input
        if not isinstance(last_n_days_data, list):
            return jsonify({'error': '"last_n_days_data" must be a list.'}), 400
        if not all(isinstance(x, (int, float)) for x in last_n_days_data):
            return jsonify({'error': 'All values in "last_n_days_data" must be numbers.'}), 400
        if any(x < 0 for x in last_n_days_data):
            return jsonify({'error': 'Values in "last_n_days_data" must not be negative.'}), 400
        if len(last_n_days_data) == 0:
            return jsonify({'error': '"last_n_days_data" must not be empty.'}), 400

        # Prediksi
        pred_expense, rec_budget = predict_next_month_expense(last_n_days_data)

        return jsonify({
            'predicted_expense': round(pred_expense, 2),
            'recommended_budget': round(rec_budget, 2)
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
