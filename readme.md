# AI Social Insights
An interactive dashboard built with Bokeh, LSTM, and Prophet to forecast social media metrics.

# About
AI Social Insights is an interactive web application designed to help you analyze and forecast social media metrics for multiple clients and platforms. Using advanced time-series forecasting techniques such as LSTM (Long Short-Term Memory) neural networks and Facebook’s Prophet model, the dashboard provides accurate future predictions, anomaly detection, AI-driven insights on metric trends (rising, dropping, or stable), and performance evaluation with RMSE (Root Mean Squared Error). This tool is ideal for marketers, analysts, and social media managers looking to optimize campaigns and understand trends in their data.

## Features
- Choose clients, platforms, and specific metrics dynamically
- Compare forecast models: LSTM and Prophet
- Visualize forecasts alongside actual historical data
- Detect anomalies in social media metrics automatically
- AI-generated insights indicating whether the selected metric is rising, dropping, or stable
- Calculate RMSE to evaluate forecasting accuracy

## How to Run

1. Clone this repo:
   git clone https://github.com/muhammadaun7/ai-social-insights.git
   cd ai-social-insights

2. Install requirements:
    pip install -r requirements.txt

3. Add your Excel files into the data/ folder. Use the following format:
| Date       | Platform  | Metric | Value |
| ---------- | --------  | ------ | ----- |
| 2025-05-01 | Facebook  | Reach  | 1234  |
| 2025-05-02 | Instagram | Reach  | 1300  |

4. Run the dashboard:
    bokeh serve --show main.py

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
