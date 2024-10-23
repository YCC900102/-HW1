from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        slope = float(request.form['slope'])
        intercept = float(request.form['intercept'])
        noise = float(request.form['noise'])
        num_points = int(request.form['num_points'])

        # 生成合成資料
        X = np.random.rand(num_points, 1) * 10  # 隨機生成 X 值
        y = slope * X + intercept + np.random.randn(num_points, 1) * noise  # 根據斜率、截距和雜訊生成 y 值

        # 訓練線性迴歸模型
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        # 計算均方誤差 (MSE)
        mse = mean_squared_error(y, y_pred)

        # 繪製資料點及迴歸線
        plt.scatter(X, y, color='blue', label='Data Points')
        plt.plot(X, y_pred, color='red', label='Regression Line')
        plt.title('Linear Regression')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()

        # 儲存圖表到一個字符串
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf8')

        return render_template('index.html', mse=mse, img_data=img_base64)

    return render_template('index.html', mse=None, img_data=None)

if __name__ == '__main__':
    app.run(debug=True)
