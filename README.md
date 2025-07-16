# 📈 Microsoft Stock Price Prediction using Machine Learning

An **interactive Streamlit web application** for **predicting Microsoft (MSFT) stock prices** using Machine Learning and Deep Learning techniques. This application applies **Bidirectional LSTM models** and technical indicators to analyze historical stock data and forecast future prices, helping users make informed, data-driven investment decisions.

🎓 **Internship Program:** AI Skills on Microsoft Azure Platform  
🏢 **Organization:** Edunet Foundation  
👩‍🎓 **Student Name:** Vani Sai Deepika  
🏫 **College Name:** Rishi MS Institute Of Engineering & Technology For Women  
💻 **Department:** Computer Science and Engineering  
✉️ **Email:** saideepikavani@gmail.com  
🧠 **AICTE Student ID:** STU6641f91d5732e1715599645

---

## 🔍 Project Objective

To design and deploy a predictive model that forecasts future Microsoft stock prices using historical stock data, technical indicators, and deep learning models.

---

## 🌻 What I Learned

This project improved my skills in using AI and machine learning for real-world applications, especially in time-series forecasting within the financial domain. I practiced working with data pipelines, model development, and deployment workflows, taking this project from an initial idea to a fully functional and deployed Streamlit web application. This has added a deployment-ready project to my portfolio, supporting my preparation for internships and future opportunities in the tech industry.

---

## 🌐 Live Demo

The project has been deployed as an interactive Streamlit web application and is available here:

[![View on Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://microsoft-stock-price-prediction-4ssuutizqupstxes6uungb.streamlit.app/)

🌻 **Check it out here:** [Click to view the deployed app](https://microsoft-stock-price-prediction-4ssuutizqupstxes6uungb.streamlit.app/)

---

## 🧠 Tech Stack Used

### 📌 Languages & Libraries 

- **Python** – Used as the primary programming language for data processing, modeling, and app development.
- **Pandas, NumPy** – For data loading, cleaning, manipulation, and efficient numerical operations on stock data.
- **Matplotlib, Seaborn, Plotly** – To create clear, interactive, and visually appealing charts for EDA and app visualization.
- **Scikit-learn** – For preprocessing (scaling), data splitting, and utility ML functions to prepare data for the model.
- **TensorFlow & Keras** – To build, train, and evaluate the Bidirectional LSTM model for time-series forecasting.
- **yFinance (for stock data retrieval)** – To fetch 10 years of historical Microsoft stock data directly within the app.

### 📌 Tools & Platforms
- **Jupyter Notebook** – For experimentation and data analysis  
- **Streamlit (VS Code)** – For running and testing the interactive web app locally

### 📌 Machine Learning & AI Methods
- **Algorithm Used**: Bidirectional Long Short-Term Memory (BiLSTM)
- MinMax Scaling – For data normalization
- Technical Indicators – RSI (Relative Strength Index), MACD (Moving Average Convergence Divergence)
- Sequence modeling – Capturing temporal patterns in stock prices
- Predictive modeling – Forecasting Microsoft stock prices for 7 to 30 future days


---

## 🗂️ Project Structure
Microsoft-Stock-Price-Prediction/      
├── app.py ← Main Streamlit app (UI logic)                           
├── requirements.txt ← Python dependencies for local deployment                    
├── README.md ← Project documentation and overview                   
├── model.pkl ← (Optional) Saved trained model                  
├── msft_notebook.ipynb ← Jupyter Notebook for experimentation                         
├── helpers/ ← Reusable ML and data processing modules                           
│ ├── init.py                                                        
│ ├── data_loader.py ← Loads stock data (e.g., from Yahoo Finance)                       
│ ├── indicators.py ← Computes RSI, MACD, and other indicators                        
│ ├── model.py ← LSTM model creation, training, prediction                       
│ ├── utils.py ← Plotting functions and helper utilities           

---

## 🚀 Features of the Web App

- ✅ Fetches 10 years of Microsoft stock data using `yfinance`
- 📊 Computes technical indicators (RSI and MACD)
- 🧠 Trains a BiLSTM model on historical stock prices
- 📈 Displays actual vs predicted prices
- 🔮 Forecasts future stock prices (1–30 days)
- 📉 Visualizations with interactive plots (Plotly)
- 📥 Option to download forecasted prices as CSV

---

## 📦 Installation & Running Locally

Follow these steps to run the Microsoft Stock Price Prediction app locally:

### 1. **Clone the repository**
```bash
git clone https://github.com/VANISAIDEEPIKA/Microsoft-Stock-Price-Prediction.git
cd Microsoft-Stock-Price-Prediction


 2. Install the required dependencies

   pip install -r requirements.txt

3. Run the Streamlit app

    streamlit run app.py
