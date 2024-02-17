from flask import Flask, request, jsonify
import pandas as pd
from prophet import Prophet

app = Flask(__name__)

# Load the retail data
df_raw = pd.read_csv("retail-usa-clothing.csv", parse_dates=['date'])

@app.route('/forecast', methods=['GET'])
def forecast():
    # Extract query parameters
    country = request.args.get('country')
    state = request.args.get('state')
    item = request.args.get('item')
    date = request.args.get('date')  # Expecting format 'YYYY-MM-DD'
    region = request.args.get('region')

    # Filter data based on the provided parameters
    # Start with a True condition to ensure all rows are included initially
    condition = (df_raw['date'].notnull())  # Assumes 'date' is always present and not null

    # Conditionally add filters
    if country is not None and country != '':
        condition &= (df_raw['country'] == country)
    if state is not None and state != '':
        condition &= (df_raw['state'] == state)
    if item is not None and item != '':
        condition &= (df_raw['item'] == item)
    if region is not None and region != '':
        condition &= (df_raw['region'] == region)

    filtered_df = df_raw[condition]


    # Ensure 'date' column is in datetime format
    filtered_df['date'] = pd.to_datetime(filtered_df['date'])

    # Apply forecasting function here; simplified for demonstration
    forecast_result = hierarchical_forecast_prophet(filtered_df, date)

    return jsonify(forecast_result)

def hierarchical_forecast_prophet(df, target_date):
    # Assuming 'date' is already in datetime format and 'quantity' is the target variable
    df_for_prophet = df.rename(columns={'date': 'ds', 'quantity': 'y'})

    model = Prophet()
    model.fit(df_for_prophet)

    future = model.make_future_dataframe(periods=120)  # Adjust periods as needed
    forecast = model.predict(future)

    # Filter forecast for the target date
    forecast_target_date = forecast[forecast['ds'] == pd.to_datetime(target_date)][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    return forecast_target_date.to_dict('records')

@app.route('/average-forecast', methods=['GET'])
def average_forecast():
    # Extract query parameters
    country = request.args.get('country')
    state = request.args.get('state')
    item = request.args.get('item')
    region = request.args.get('region')
    from_date = request.args.get('from_date')
    to_date = request.args.get('to_date')

    # Apply the same filtering based on the provided parameters
    condition = (df_raw['date'].notnull())
    if country:
        condition &= (df_raw['country'] == country)
    if state:
        condition &= (df_raw['state'] == state)
    if item:
        condition &= (df_raw['item'] == item)
    if region:
        condition &= (df_raw['region'] == region)

    filtered_df = df_raw[condition]

    # Call a function to perform forecasting and calculate the average forecast value
    average_forecast_result = calculate_average_forecast(filtered_df, from_date, to_date)

    return jsonify(average_forecast_result)

def calculate_average_forecast(df, from_date, to_date):
    # Prepare the DataFrame for Prophet
    df_for_prophet = df.rename(columns={'date': 'ds', 'quantity': 'y'})
    df_for_prophet.to_csv("Check.csv")
    
    # Initialize and fit the Prophet model
    model = Prophet()
    model.fit(df_for_prophet)

    # Create future dates DataFrame starting from the last historical date up to 'to_date'
    last_historical_date = df_for_prophet['ds'].max()
    future_dates = model.make_future_dataframe(periods=(pd.to_datetime(to_date) - last_historical_date).days + 1)
    
    # Predict over the future dates
    forecast = model.predict(future_dates)
    
    # Filter the forecast for the specified date range and calculate the average of the 'yhat' values
    forecast_filtered = forecast[(forecast['ds'] >= pd.to_datetime(from_date)) & (forecast['ds'] <= pd.to_datetime(to_date))]
    average_forecast = forecast_filtered['yhat'].mean()
    
    # Return the average forecast value
    return {
        'from_date': from_date,
        'to_date': to_date,
        'average_forecast': average_forecast
    }





if __name__ == '__main__':
    app.run(debug=True)
