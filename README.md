### Introduction

### Stock Forecasting in SE

This project involves feature engineering, analyzing and forecasting stock of real equipment in SE remains using data preprocessing and machine learning techniques. The goal is to clean the data, analyze it, and then use different regression models to forecast, which items are best sold and should be kept in stock.

### Usage

Clone the repository:
```bash
git clone https://github.com/CVliner/se_keep_stock_items.git
```
```bash
cd stock-forecasting
```
Place the /Data/Stock_forecast.csv dataset in the project directory and run the script:
```bash
python stock_forecasting.py
```

### Dataset
Datas can be found in /Data/Stock_forecast.csv. Dataset has been collected according to purchase of SE equipment during whole year in 2018.
### Results
From the given diagrams, it is evident that the most purchased items in KZ by customers are electrical frames, vacuum switchers (Evolis), Micom controllers, and service jobs. Conversely, the least sold items are specialized devices like maintenance plates, connecting cables, and other spare parts, used mainly in specific projects. Using a regression model to forecast future sales, evaluated by RMSE and R² metrics, it is recommended to store popular items locally. This will reduce delivery time and significantly enhance sales efficiency and business operations.
### Recommendations
For further observation and SE stock remainings prediction, it is recommended to:

- Update the model and improve prediction models.
- Continue data analysis with addition equipment for last 5 years.
- Create list of continously purchased equipment in SE and keep them in warehouse.

