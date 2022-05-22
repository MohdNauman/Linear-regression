

import pandas as pd
import numpy as np


data = pd.read_csv(r'D:\Data science\Simple leaner regression\emp_data.csv')

data.describe()


import matplotlib.pyplot as plt

plt.bar(height = data.Churn_out_rate, x = np.arange(1, 11, 1))
plt.hist(data.Churn_out_rate) 
plt.boxplot(data.Churn_out_rate) 

plt.bar(height = data.Salary_hike, x = np.arange(1, 11, 1))
plt.hist(data.Salary_hike) 
plt.boxplot(data.Salary_hike)


plt.scatter(x = data['Salary_hike'], y = data['Churn_out_rate'], color = 'green') 


np.corrcoef(data.Salary_hike, data.Churn_out_rate)


cov_output = np.cov(data.Salary_hike, data.Churn_out_rate)[0, 1]
cov_output


import statsmodels.formula.api as smf


model = smf.ols('Churn_out_rate ~ Salary_hike', data = data).fit()
model.summary()


pred1 = model.predict(pd.DataFrame(data['Salary_hike']))

# Regression Line
plt.scatter(data.Salary_hike, data.Churn_out_rate)
plt.plot(data.Salary_hike, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = data.Churn_out_rate - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

######## Model 2

plt.scatter(x = np.log(data['Salary_hike']), y = data['Churn_out_rate'], color = 'brown')

np.corrcoef(np.log(data.Salary_hike), data.Churn_out_rate)

model2 = smf.ols('Churn_out_rate ~ np.log(Salary_hike)', data = data).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(data['Salary_hike']))

plt.scatter(np.log(data.Salary_hike), data.Churn_out_rate)
plt.plot(data.Salary_hike, pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

res2 = data.Churn_out_rate - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2
 

####### Model 3

plt.scatter(x = data['Salary_hike'], y = np.log(data['Churn_out_rate']), color = 'green') 

np.corrcoef(data.Salary_hike, np.log(data.Churn_out_rate))

model3 = smf.ols('np.log(Churn_out_rate) ~ Salary_hike', data = data).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(data['Salary_hike']))


plt.scatter(data.Salary_hike, np.log(data.Churn_out_rate))
plt.plot(data.Salary_hike, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

res3 = data.Churn_out_rate - pred3
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3

#######model 4

model4 = smf.ols('np.log(Churn_out_rate) ~ Salary_hike + I(Salary_hike*Salary_hike)', data = data).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(data))
pred4_at = np.exp(pred4)
pred4_at


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = data.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)

res4 = data.Churn_out_rate - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4


data1 = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data1)
table_rmse


from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size = 0.2)

plt.scatter(train.Salary_hike, train.Churn_out_rate)

plt.figure(2)
plt.scatter(test.Salary_hike, test.Churn_out_rate)


finalmodel = smf.ols('np.log(Churn_out_rate) ~ Salary_hike + I(Salary_hike*Salary_hike)', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_Churn_out_rate = np.exp(test_pred)
pred_test_Churn_out_rate

# Model Evaluation on Test data
test_res = test.Churn_out_rate - pred_test_Churn_out_rate
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_Churn_out_rate = (train_pred)
pred_train_Churn_out_rate

# Model Evaluation on train data
train_res = train.Churn_out_rate - pred_train_Churn_out_rate
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse






































