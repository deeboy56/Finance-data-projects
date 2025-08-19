import kaggle as kg 
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from  datetime import datetime
import scipy  as sc
import scipy.stats as stats
from scipy.stats import kurtosis 
from scipy.stats import skew
from scipy.stats import norm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import het_arch
from arch import arch_model

import seaborn as sns

kg.api.authenticate()
kg.api.dataset_download_files('novandraanugrah/xauusd-gold-price-historical-data-2004-2024', path='.', unzip=True)
kg.api.dataset_metadata('novandraanugrah/xauusd-gold-price-historical-data-2004-2024', path='.')


D_XAU=pd.read_csv(r"C:\Users\dnyla\xauusd-gold-price-historical-data-2004-2024\XAU_1d_data_2004_to_2024-09-20.csv")


#Combinging datatime o filter and sort 
D_XAU["Datetime"]=pd.to_datetime(D_XAU["Date"]+" "+D_XAU["Time"], format="%Y.%m.%d %H:%M")

d_XAU=D_XAU.drop(columns=["Date","Time"])

#filter data for year
DfDaily = d_XAU.loc[(d_XAU['Datetime'].dt.year >= 2023) & (d_XAU['Datetime'].dt.year <= 2024)]

#creating and plotting log returns
DfDaily["return"]=np.log(DfDaily['Close']/DfDaily['Open'])
DfDaily["Pips"]=DfDaily["return"]*10000
plt.figure(figsize=(30,30))
plt.legend()
plt.plot(DfDaily.index,DfDaily['return'],label='Gold Daily returns',color='green')

#printing kurtosis to see if distributions fit IID conditions
print(kurtosis(DfDaily['return'],axis=0, bias=True))
print(skew(DfDaily['return'],axis=0, bias=True))
#ploting the acual return disttributtion
sns.displot(DfDaily,x='return',kind="kde",bw_adjust=.7,color='black',label='Gold returns')
mean = np.mean(DfDaily['return'])
std = np.std(DfDaily['return'])
#calculating the estimated student t distribution
params = stats.t.fit(DfDaily["return"])
df, loc, scale = params
x1 = np.linspace(min(DfDaily["return"]), max(DfDaily["return"]), 731)
pdf1 = stats.t.pdf(x1, df, loc, scale)
plt.plot(x1, pdf1, 'r', label=f"Fitted t-distribution\n$\mu={loc:.2f}, \sigma={scale:.2f}, df={df:.2f}$",color="pink")
#fiting normal distribution and student t
data= np.random.normal(mean,std,733)
mu_est, sigma_est = norm.fit(data)
x = np.linspace(min(data), max(data), 731)
pdf = norm.pdf(x, mu_est, sigma_est)
plt.plot(x, pdf, 'r', label=f"Normal Fit\n$\mu={mu_est:.2f}, \sigma={sigma_est:.2f}$",color='orange')
plt.legend()
plt.title("Normal Distribution Estimate and Fitted t")
plt.xlabel("Value")
plt.ylabel("Density")
plot_pacf(DfDaily['return'])
plt.show()

#autocorrelaion testing
arch_test = het_arch(DfDaily['return'])
print(f"P-value: {arch_test[1]}")
DfDaily['return'].rolling(window=30).std().plot(figsize=(10, 5))
plt.title("Rolling 30-day Standard Deviation of Log Returns")
plt.show()

# Extract return series
returns = DfDaily["return"]

# Set rolling window size
rolling_window = 252  # 1 year of daily data

# Initialize list to store forecasts
vol_forecasts = []

# Initialize list to store corresponding dates
forecast_dates = []

# Rolling estimation loop
for i in range(rolling_window, len(returns)):
    train_data = returns[:i]  # Use data up to the current point
    model = arch_model(train_data, vol="Garch", p=1, q=1, dist="t")  # Fit EGARCH(1,1)
    fit = model.fit(disp="off")  # Suppress output
    fit.params["nu"] = 5.87
    forecast = fit.forecast(horizon=1)  # 1-day-ahead forecast
    print(fit)
    # Store forecasted volatility (sqrt of variance)
    vol_forecasts.append(np.sqrt(forecast.variance.iloc[-1, 0]))

    # Store the date corresponding to the forecast
    forecast_dates.append(returns.index[i])  

# Create DataFrame with rolling volatility forecasts
rolling_vol_df = pd.DataFrame({
    "Date": forecast_dates,
    "Rolling_GARCH_Volatility": vol_forecasts
})

# Set the Date column as the index
rolling_vol_df.set_index("Date", inplace=True)

# Display the first few rows
print(rolling_vol_df.head())


plt.figure(figsize=(12, 6))
plt.plot(rolling_vol_df, label="Rolling GARCH(1,1) Forecasted Volatility", color="purple")
plt.title("1-Day Ahead Rolling Volatility Forecast")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.legend()
plt.show()

rolling_vol_df['95% range']=rolling_vol_df['Rolling_GARCH_Volatility']*2.571