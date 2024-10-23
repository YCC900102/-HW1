import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from datetime import datetime

# 1. 讀取 CSV 檔案
file_path = r"C:\Users\EE715\Documents\HW2test\2330-training.csv"
df = pd.read_csv(file_path)

# 2. 將 Date 欄位轉換為日期格式
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

# 3. 移除 y 欄位中的逗號並轉換為 float
df['y'] = df['y'].str.replace(',', '').astype(float)

# 4. 準備數據給 Prophet
df.rename(columns={'Date': 'ds'}, inplace=True)

# 5. 使用 Prophet 模型進行預測
model = Prophet(changepoint_prior_scale=0.5, yearly_seasonality=False, weekly_seasonality=False)
model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
model.fit(df)

# 6. 預測未來 60 天
future = model.make_future_dataframe(periods=60)
forecast = model.predict(future)

# 7. 繪圖
plt.figure(figsize=(14, 7))

# 歷史數據
plt.plot(df['ds'], df['y'], color='black', label='Historical Data')

# 預測數據
plt.plot(forecast['ds'], forecast['yhat'], color='purple', label='Forecast')

# 不確定性區間
plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='lightblue', alpha=0.4)

# 歷史平均線
plt.axhline(y=df['y'].mean(), color='blue', linestyle='--', label='Historical Average')

# 添加紅色箭頭指向最後數據點
last_data_point = df['y'].iloc[-1]
plt.annotate('Last Data Point', 
             xy=(df['ds'].iloc[-1], last_data_point), 
             xytext=(df['ds'].iloc[-1], last_data_point + 50),
             arrowprops=dict(facecolor='red', shrink=0.05),
             fontsize=12, ha='center')

# 添加綠色箭頭指向預測區間的下緣
downward_point = forecast['yhat_lower'].iloc[-1]
plt.annotate('Downward Point', 
             xy=(forecast['ds'].iloc[-1], downward_point), 
             xytext=(forecast['ds'].iloc[-1], downward_point - 50),
             arrowprops=dict(facecolor='green', shrink=0.05),
             fontsize=12, ha='center')

# 調整日期格式和標籤
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)

# 圖例
plt.legend(loc='upper right')

# 調整布局
plt.tight_layout()
plt.title('Stock Price Prediction using Prophet')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.grid()
plt.show()
