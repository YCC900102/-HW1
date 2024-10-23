from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            df = pd.read_csv(file)
            df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
            df['y'] = df['y'].str.replace(',', '').astype(float)
            df.rename(columns={'Date': 'ds'}, inplace=True)

            model = Prophet(changepoint_prior_scale=0.5, yearly_seasonality=False, weekly_seasonality=False)
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            model.fit(df)

            future = model.make_future_dataframe(periods=60)
            forecast = model.predict(future)

            plt.figure(figsize=(14, 7))
            plt.plot(df['ds'], df['y'], color='black', label='Historical Data')
            plt.plot(forecast['ds'], forecast['yhat'], color='purple', label='Forecast')
            plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='lightblue', alpha=0.4)
            plt.axhline(y=df['y'].mean(), color='blue', linestyle='--', label='Historical Average')

            last_data_point = df['y'].iloc[-1]
            plt.annotate('Last Data Point', 
                         xy=(df['ds'].iloc[-1], last_data_point), 
                         xytext=(df['ds'].iloc[-1], last_data_point + 50),
                         arrowprops=dict(facecolor='red', shrink=0.05),
                         fontsize=12, ha='center')

            downward_point = forecast['yhat_lower'].iloc[-1]
            plt.annotate('Downward Point', 
                         xy=(forecast['ds'].iloc[-1], downward_point), 
                         xytext=(forecast['ds'].iloc[-1], downward_point - 50),
                         arrowprops=dict(facecolor='green', shrink=0.05),
                         fontsize=12, ha='center')

            plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.title('Stock Price Prediction using Prophet')
            plt.xlabel('Date')
            plt.ylabel('Stock Price')
            plt.grid()

            # 保存圖表到 static 文件夾
            image_path = os.path.join('static', 'forecast.png')
            plt.savefig(image_path)
            plt.close()

            return render_template('index.html', image=image_path)

    return render_template('index.html', image=None)

if __name__ == '__main__':
    app.run(debug=True)