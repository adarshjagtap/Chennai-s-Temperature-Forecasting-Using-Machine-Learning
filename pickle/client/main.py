from flask import Flask, request, render_template
from datetime import datetime, timedelta
import pickle

app = Flask(__name__)

@app.route("/")
def root():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def Forecast():
    # Parse the form data
    current_date_str = request.form['EnterDate']
    days_to_forecast = int(request.form['NumDays'])

    # Convert current_date_str to datetime object
    try:
        current_date = datetime.strptime(current_date_str, '%Y-%m-%d')  # Assuming the date format is YYYY-MM-DD
    except ValueError:
        return "Invalid date format. Please use YYYY-MM-DD.", 400

    # Load the model from model.pkl file
    try:
        with open('./model.pkl', 'rb') as file:
            model = pickle.load(file)
    except FileNotFoundError:
        return "Model file not found.", 500
    except pickle.UnpicklingError:
        return "Error loading the model.", 500

    # Forecast future values
    try:
        # Use the `get_forecast()` method for ARIMA models
        forecast = model.get_forecast(steps=days_to_forecast)
        forecast_mean = forecast.predicted_mean.tolist()
    except Exception as e:
        return f"Error making prediction: {str(e)}", 500

    # Generate a list of dates corresponding to the forecast
    forecast_dates = [(current_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days_to_forecast)]

    # Combine dates and forecast into a list of tuples
    forecast_data = list(zip(forecast_dates, forecast_mean))

    # Render the result
    return render_template('result.html', forecast_data=forecast_data)

# Start the application
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4000, debug=True)
