import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from google.colab import files


file= files.upload()
df= pd.read_csv("latr.csv")
df.head(15)


df.columns
df= df.dropna()
df= df.drop(['SI.NO', 'Date','Time','Moon_P'], axis=1)
df.head(5)

df["R_Tide"].mean()

plt.boxplot(df.R_Tide)
plt.show()

plt.hist(df.R_Tide,facecolor='purple',alpha=0.5)
plt.show()

p=df["R_Tide"].skew() 
print("skew is",p)
q=df["R_Tide"].kurt()
print("Kurtosis is",q)

print("correlation between range of tides and moon distance",df.R_Tide.corr(df.Moon_D) )
print("correlation between range of tides and moon's gravitational pull",df.R_Tide.corr(df.Moon_G))
print("correlation between range of tides and sun distance",df.R_Tide.corr(df.Sun_D))
print("correlation between range of tides and temperature",df.R_Tide.corr(df.Temp))
print("correlation between range of tides and dew point",df.R_Tide.corr(df.Dew_Pt))
print("correlation between range of tides and humidity",df.R_Tide.corr(df.Humi))
print("correlation between range of tides and pressure",df.R_Tide.corr(df.Pres))
print("correlation between range of tides and precipitation",df.R_Tide.corr(df.Precip))


plt.plot(df.R_Tide,df.Moon_D,"ro")
plt.xlabel("range")
plt.ylabel("Moon dist")
plt.show()

plt.plot(df.R_Tide,df.Moon_G,"go")
plt.xlabel("range")
plt.ylabel("Moon Grav")
plt.show()


plt.plot(df.R_Tide,df.Sun_D,"bo")
plt.xlabel("range")
plt.ylabel("sun dist")
plt.show()


plt.plot(df.R_Tide,df.Temp,"yo")
plt.xlabel("range")
plt.ylabel("temp")
plt.show()


plt.plot(df.R_Tide,df.Dew_Pt,"mo")
plt.xlabel("range")
plt.ylabel("dew")
plt.show()


plt.plot(df.R_Tide,df.Humi,"mo")
plt.xlabel("range")
plt.ylabel("humi")
plt.show()


plt.plot(df.R_Tide,df.Pres,"go")
plt.xlabel("range")
plt.ylabel("pressure")
plt.show()


plt.plot(df.R_Tide,df.Precip,"bo")
plt.xlabel("range")
plt.ylabel("precip")
plt.show()


sns.pairplot(df)
plt.savefig("df.jpg")


X = df.loc[:, ['Moon_D','Sun_D','Temp','Dew_Pt','Humi','wind_Sp','Pres','Precip']]
y = df['R_Tide']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model=sm.OLS(y_train,X_train).fit()
model.summary()


model_uncentered = LinearRegression().fit(X_train, y_train)
r_squared_uncentered = model_uncentered.score(X_train, y_train)
mean_y = np.mean(y_train)
y_centered = y_train - mean_y
model_centered = LinearRegression().fit(X_train, y_centered)
r_squared_centered = model_centered.score(X_train, y_centered)
print("Uncentered R-squared:", r_squared_uncentered)
print("Centered R-squared:", r_squared_centered)


y_pred= model.predict(X_test)
y_pred
Plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, label='Predicted vs Actual')
plt.plot(y_test, y_test, color='red', label='Best Fit Line')
plt.xlabel('Actual R_Tide')
plt.ylabel('Predicted R_Tide')
plt.title('Actual vs Predicted R_Tide with Best Fit Line')
plt.legend()
plt.show()


