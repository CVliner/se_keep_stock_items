import pandas as pd
import numpy as np
import requests
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Libraries imported, if imported correctly the below will print
print("Libraries imported.")

"""Upload data set"""

# Data set upload
filepath = "/Data/Stock_forecast.csv"

# First look at the dataset
df = pd.read_csv(filepath, encoding = "cp1251") # Encoding as would not open
df.head(1)

# Initial look at the data types
# df.info()

# Turns columns to datetime data type
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Seperate date column
df["Date"] = df["InvoiceDate"].dt.date
# Time column
df["InvoiceTime"] = df["InvoiceDate"].dt.time
# df.info()

# convert column from object to string
df["Description"] = df["Description"].astype(str)

# Storing rows with words found throw the project that are not product sales
remove_words = ["Discount", "SAMPLES"]

# Boolean indexing to drop rows containing specific words
df = df[~df["Description"].str.contains("|".join(remove_words))]

df.isnull().sum()

# Have a total order column by multiplying the quantity by unit price
df["TotalPrice"]= df["Quantity"]*df["Price"]

df.head(5)

# Missing values
# df.isnull().sum()

# Checking the df
# df.describe()

# Assuming that minus values are refunds
# print(df[df["Quantity"] < 0])

# Need to reference/ If quantity is negative, it is a returned or cancelled item
df["Transaction Type"] = np.where(df["Quantity"] < 0, "Cancelled", "Sale")

# Print the resulting dataframe
# print(df)

#@title Duplicates
# If row duplicated store here
Duplicates = df[df.duplicated(keep=False)]

# If statement which shows
if len(Duplicates) > 0:
  # Prints the length or count of duplicate rows
    print("Total count of duplicate rows:", len(Duplicates))
# If all duplicates missing or now deleted will show no duplicate found
else:
    print("No duplicate rows found.")


# Drop doubles
df = df.drop_duplicates()

# To check if deleted now
Duplicates2 = df[df.duplicated(keep=False)]

# If statement which shows
if len(Duplicates2) > 0:
  # Prints the length or count of duplicate rows
    print("Total count of duplicate rows:", len(Duplicates2))
# If all duplicates missing or now deleted will show no duplicate found
else:
    print("No duplicate rows found, they have now been deleted.")

#@title Start/End Date
# Looking for start and end date in df
StartDate = (df["InvoiceDate"].min())
EndDate = (df["InvoiceDate"].max())
print("Start date of data set: ", StartDate)
print("End date of data set: ", EndDate)

df.describe()

# Locating outliers throuhg a box plot, on column qunatity by price
df.loc[:, ["Quantity", "TotalPrice"]].boxplot(figsize = (10,10));

# Function with some per arguments inside
def remove_outliers(data, lower_percentile = 0.25, upper_percentile = 0.75):
  # Defining Q's
    q1 = data.quantile(lower_percentile)
    q3 = data.quantile(upper_percentile)
    # Finding the inter quartile range
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    # Checks if data is in side the quartile range and stored
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    return filtered_data

# Outliers removed for quantity
df.Quantity = remove_outliers(df.Quantity)
# Outliers removed for price
df.Revenue = remove_outliers(df.TotalPrice)
df.dropna(inplace=True)
df.loc[:, ["Quantity", "TotalPrice"]].describe()

"""# Data Analysis (Product)"""

#@title Country Sales
import plotly.express as px

# New Country df with columns for the graph
CountryDF = df[["Region", "Quantity", "TotalPrice"]]

# Grouping all countries to see the biggest buyers with.sum()
CountryDFgrouped = CountryDF.groupby("Region").sum()

# Top 5 countries based on Quantity
CountryDFquantity = CountryDFgrouped.sort_values("Quantity", ascending = False).head(20)

# Bar chart for Quantity on x and country on y axis using Plotly made horizontal
fig_quantity = px.bar(CountryDFquantity, x = "Quantity", y = CountryDFquantity.index,
                      orientation = "h", labels={"Quantity": "Quantity Sold", "y": "Region"}, title = "Top 20 Countries by Quantity Sold")

# Adding in colour code for graph
fig_quantity.update_traces(marker = dict(color = px.colors.sequential.Oranges))

fig_quantity.show()

# top 5 countries based on Total Price
CountryDFprice = CountryDFgrouped.sort_values("TotalPrice", ascending = False).head(20)

# Bar chart for Total Price on x and country on y axis using Plotly made horizontal
fig_price = px.bar(CountryDFprice, x = "TotalPrice", y = CountryDFprice.index, orientation="h",
                   labels={"TotalPrice": "Total Price", "y": "Region"}, title = "Top 20 Countries by Total Price")

# Adding in colour code for graph
fig_price.update_traces(marker = dict(color = px.colors.sequential.Oranges))

fig_price.show()

#@title Best/Worst selling products
# Group the data by product and sum the total sales, quantity, and calculate the unit price for each product
ProductSummary = df.groupby("Description").agg({"TotalPrice": "sum", "Quantity": "sum"})
ProductSummary["Price"] = ProductSummary["TotalPrice"] / ProductSummary["Quantity"]

# Sort the products by total sales in descending order and select the top 10
BestProduct = ProductSummary.sort_values(by="TotalPrice", ascending=False).head(10)

# Print the top 10 products
print("Top 10 best-selling products:")
print(BestProduct)

# Plot the top 10 products and their sales
sns.barplot(x = BestProduct.index, y = "TotalPrice", data = BestProduct)
plt.title("Top 10 best-selling products")
plt.xlabel("Product")
plt.ylabel("Total Sales")
plt.xticks(rotation=90)
plt.show()

# Sort the products by total sales in ascending order and select the worst 10
WorstProduct = ProductSummary.sort_values(by="TotalPrice").head(100)

# Print the worst 10 products
print("Worst 10 products:")
# Prints in list format
print(WorstProduct)

# Plot the worst 10 products and their sales
sns.barplot(x = WorstProduct.index, y  ="TotalPrice", data = WorstProduct)
# Addint titles to the graoh
plt.title("Worst 10 products by sales")
plt.xlabel("Product")
plt.ylabel("Total Sales")
plt.xticks(rotation=90)
plt.show

#@title Best/ Worst selling products
import plotly.graph_objects as go
import plotly.express as px

# Sort the products by total sales in descending order and select the top 10
BestProduct = ProductSummary.sort_values(by = "TotalPrice", ascending = False).head(10)

# Plot the top 10 products and their sales
fig = px.bar(BestProduct, x=BestProduct.index, y = "TotalPrice", hover_data = ["Quantity", "Price"])
fig.update_layout(
    title="Top 10 best-selling products",
    xaxis_title="Product",
    yaxis_title="Total Sales")

# Adding in colour code for graoh
fig.update_traces(marker=dict(color=px.colors.sequential.Greens))

fig.show()

# Sort the products by total sales in ascending order and select the worst 10
WorstProduct = ProductSummary.sort_values(by = "TotalPrice").head(10)

# Plot the worst 10 products and their sales
fig = px.bar(WorstProduct, x = WorstProduct.index, y="TotalPrice", hover_data=["Quantity", "Price"])
fig.update_layout(title = "Worst 25 products by sales", xaxis_title = "Product", yaxis_title = "Total Sales")

# Adding in colour code for graph

fig.show()

# Create own month and year column with pandas to see if month has any correlation with the data set
df["Year"] = pd.to_datetime(df["InvoiceDate"]).dt.year
df["Quarter"] = df.InvoiceDate.dt.quarter
df["Month"] = pd.to_datetime(df["InvoiceDate"]).dt.month
df["Week"] = pd.to_datetime(df["InvoiceDate"]).dt.isocalendar().week

# Convert InvoiceDate to datetime type
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Create a new column IsWeekend based on the day of the week
df["IsWeekend"] = (df["InvoiceDate"].dt.dayofweek >= 5).astype(int)

# To check it worked, uses bool to check if 1 which means yes it is weekend
weekend = df[df["IsWeekend"] == True]
# Check it worked
print(weekend)

#@title Duplicates
# If row duplicated store here
Duplicates = df[df.duplicated(keep=False)]

if len(Duplicates) > 0:
  # Prints the length or count of duplicate rows
  print("Current count of duplicate rows:", len(Duplicates))
  # Drop doubles
  df = df.drop_duplicates()
# If all duplicates missing or now deleted will show no duplicate found
else:
  print("No duplicate rows found to begin with.")

# If there are still duplicated rows
Duplicates2 = df[df.duplicated(keep=False)]
if len(Duplicates2) > 0:
  # Prints the length or count of duplicate rows
  print("Total count of duplicate rows after current count:", len(Duplicates2))
# If all duplicates missing or now deleted will show no duplicate found
else:
  print("No duplicate rows found, all have been deleted successfully.")

"""# Encoding Data

Encoding to numeric/binary value
"""

df.info()
PreDf = df.copy()
PreDf.info()

from sklearn.preprocessing import LabelEncoder

# Enabelling the LabelEncoder
le = LabelEncoder()

# apply label encoding to the columns
df["Stock Code"] = le.fit_transform(df["Stock Code"])
df["Description"] = le.fit_transform(df["Description"])
df["Region"] = le.fit_transform(df["Region"])
df["Customer"] = le.fit_transform(df["Customer"])
df["Transaction Type"] = le.fit_transform(df["Transaction Type"])

# Sorting data types
df["Year"] = df["InvoiceDate"].dt.year
df["Date"] = df["InvoiceDate"].dt.day
# df["Week"] = df["InvoiceDate"].dt.week
df["InvoiceTime"] = df["InvoiceDate"].dt.hour * 60 + df["InvoiceDate"].dt.minute

# Make Range between Max and min
AllDates = pd.date_range(start = StartDate, end = EndDate, freq = "D").date

# Dates between max and min
IncludedDates = df["InvoiceDate"].dt.date.unique()

# Dates not included in df
MissingDates = np.setdiff1d(AllDates, IncludedDates)

# All Missing Dates
print("Dates where there were no invoices sold:\n\n", MissingDates)

df.info()

"""# Data Anlaysis (Sales)"""

# Compute the total monthly sales
WeeklySales = df.groupby(["Year", "Week"])["Quantity"].sum().reset_index()

# Plot the historical monthly sales with year hue
plt.figure(figsize=(10, 5))
sns.lineplot(x="Week", y="Quantity", hue="Year", data=WeeklySales)
plt.xlabel("Week")
plt.ylabel("Total Sales")
plt.title("Historical Weekly Sales")
plt.legend(title="Year")
plt.show()

# Compute the total monthly sales
MonthlySales = df.groupby(["Year", "Month"])["Quantity"].sum().reset_index()

# Plot the historical monthly sales with year hue
plt.figure(figsize=(10, 5))
sns.lineplot(x="Month", y="Quantity", hue ="Year", data=MonthlySales)
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.title("Historical Monthly Sales")
plt.legend(title="Year")
plt.show()

"""# Train Test Split"""

from sklearn.model_selection import train_test_split

# Dropping as no needed
X = df.drop(["InvoiceDate"],axis=1)

# Target Variable, predict sales
y = df.TotalPrice
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, shuffle=False)
print(X_train)
X_test.info()

"""# ML Algorithm (Inventory Forecasting)


"""

df.isnull().sum()

# Import necessary libraries
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Trained and evaluated different regression models
models = [LinearRegression(), DecisionTreeRegressor(), XGBRegressor()]

# For loop to print over each model
for model in models:
    # Train
    model.fit(X_train, y_train)
    # Predict
    y_pred = model.predict(X_test)
    # Check accuracy RMSE and R^2 scores
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)
    print(f"{type(model).__name__}: RMSE={rmse}, R^2={r2}")

    # Calculate the forecasted values
    if isinstance(model, LinearRegression):
        forecast = model.predict(X_test)

    # Plot predicted vs actual valuess
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual values")
    plt.ylabel("Predicted values")
    # Title wiht model name
    plt.title(f"{type(model).__name__} Predicted vs Actual Values")

    # Add a line diagonally to aid visulisation
    min_value = min(y_test.min(), y_pred.min())
    max_value = max(y_test.max(), y_pred.max())
    plt.plot([min_value, max_value], [min_value, max_value], "r--")
    plt.show()

# library needed
from plotly.subplots import make_subplots

# Function to show graph
def ShowMe(date, true, preds):
    # Plots
    fig = make_subplots(rows=1, cols=1)

    # Adding the true values to the graph
    fig.add_trace(go.Scatter(x = date, y = true.iloc[:, 0], mode = "lines", marker = dict(color = "#783242"), name = "True"))
    # Addes forcasted values to the graph
    fig.add_trace(go.Scatter(x = date, y = preds.iloc[:, 0], mode="lines", name = "Preds"))

    # Tidying up the graph
    fig.update_layout(
        xaxis = dict(title = "Date"),
        yaxis = dict(title = "TotalPrice"),
        title = "Forecasted Values vs True Values"
    )

    fig.show()

import plotly.graph_objects as go

# If else codnition for graph with forcasted values on top of actual
if isinstance(model, LinearRegression):
    forecast = model.predict(X_test.drop("forecast", axis=1))
else:
    forecast = model.predict(X_test)

# Created a new DataFrame for the forecasted values
ForecastDF = pd.DataFrame({"Date": X_test["Date"], "forecast": forecast})

# Combine the forecasted values with the historical data
CombinedDF = pd.concat([df, ForecastDF], ignore_index=True)

# Split the combined DataFrame into true and predicted values based on the splitter index
splitter = round(len(df) * 0.75)
true = CombinedDF.loc[splitter:, ["Date", "TotalPrice"]].groupby("Date").mean()
preds = CombinedDF.loc[splitter:, ["Date", "forecast"]].groupby("Date").mean()

# Plot the true and predicted values
ShowMe(true.index, true, preds)

PreDf.info()
