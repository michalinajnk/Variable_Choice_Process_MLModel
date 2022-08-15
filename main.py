import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import sns as sns
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np


# import statsmodels.tools.tools.add_constant

df = pd.read_csv('auto.csv')
print(df)

df2 = df[df.horsepower != '?']

print(df2)

plt.scatter(df.cylinders, df.mpg)
plt.xlabel('Cylinders')
plt.ylabel('Mpg')
plt.show()

plt.scatter(df.displacement, df.mpg)
plt.xlabel('Displacement')
plt.ylabel('Mpg')
plt.show()

plt.scatter(df.horsepower, df.mpg)
plt.xlabel('Horsepower')
plt.ylabel('Mpg')
plt.show()

plt.scatter(df.weight, df.mpg)
plt.xlabel('Weight')
plt.ylabel('Mpg')
plt.show()

plt.scatter(df.acceleration, df.mpg)
plt.xlabel('Acceleration')
plt.ylabel('Mpg')
plt.show()

df2['horsepower'] = df2['horsepower'].astype(int)
df2['mpg'] = df2['mpg'].astype(int)
x = df2['horsepower']
y = df2['mpg']
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
summary = model.summary()
print(summary)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 / 3, random_state=0)
rg = LinearRegression()

rg.fit(x_train, y_train)
err = []

err.append(r2_score(y_test, rg.predict(x_test)))  # test MSE
err.append(r2_score(y_train, rg.predict(x_train)))  # train MSE
plt.barh(['Test', 'Train'], err)
for i, v in enumerate(err):
    plt.text(v, i, str(np.round(v, 3)))
plt.ylabel("R^2")
plt.suptitle("Linear Regression R^2 for Test and Train Data")
plt.show()

y_pred = rg.predict(x_test)
x_pred = rg.predict(x_train)

x_train = np.arange(0, len(x_train), 1)

plt.scatter(x_train, y_train, color="green")
plt.plot(x_train, x_pred, color="red")
plt.title("Linear regression of mpg value based on a horsepower variable")
plt.xlabel("horsepower")
plt.ylabel("mpg")
plt.show()
##################################################################################
plot = seaborn.pairplot(df2)
print(plot)

################EXAMINE COORELATION################################################
"""Correlation Regression Analysis is a technique through 
which we can detect and analyze the relationship between
 the independent variables as well as with the target value."""

"""Using Correlation analysis, we can detect 
the redundant variables i.e.the variables that represent 
the same information for the target value.
"""

"""
Correlation matrix, the relationship 
between variables is a value between range -1 to +1.
"""

"""
if the viariables are higly correlated iit is a sign to drop one of this varaibles, because
the depict the same information
"""

df2 = df2.select_dtypes(np.number)
colum_names= [df2.colmuns.names()]
corr = df2.loc[:, colum_names].corr()
print(corr)







########################################################################################
x= df2[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration']]
y = df2['mpg']
#This parameter controls the shuffling applied to the data before applying the split.
# Pass an int for reproducible output across multiple function calls.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 / 3, random_state=100)
mrl = LinearRegression()

rg.fit(x_train, y_train)
err = []

plt.title("All variable taken into account ")
err.append(r2_score(y_test, rg.predict(x_test)))  # test MSE
err.append(r2_score(y_train, rg.predict(x_train)))  # train MSE
plt.barh(['Test', 'Train'], err)
for i, v in enumerate(err):
    plt.text(v, i, str(np.round(v, 5)))
plt.ylabel("R^2")
plt.suptitle("Linear Regression R^2 for Test and Train Data")
plt.show()

#######################################################################################################
x = df2[['horsepower', 'weight', 'acceleration']]
y = df2['mpg']
#This parameter controls the shuffling applied to the data before applying the split.
# Pass an int for reproducible output across multiple function calls.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 / 4, random_state=100)
rg = LinearRegression()

rg.fit(x_train, y_train)
err = []

plt.title("variables - 'horsepower', 'weight', 'acceleration' ")
err.append(r2_score(y_test, rg.predict(x_test)))  # test MSE
err.append(r2_score(y_train, rg.predict(x_train)))  # train MSE
plt.barh(['Test', 'Train'], err)
for i, v in enumerate(err):
    plt.text(v, i, str(np.round(v, 5)))
plt.ylabel("R^2")
plt.suptitle("Linear Regression R^2 for Test and Train Data")
plt.show()
###########################################################################
x = df2[['cylinders', 'displacement', 'horsepower']]
y = df2['mpg']
#This parameter controls the shuffling applied to the data before applying the split.
# Pass an int for reproducible output across multiple function calls.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 / 4, random_state=100)
rg = LinearRegression()

rg.fit(x_train, y_train)
err = []

plt.title("variables - 'cylinders', 'displacement', 'horsepower' ")
err.append(r2_score(y_test, rg.predict(x_test)))  # test MSE
err.append(r2_score(y_train, rg.predict(x_train)))  # train MSE
plt.barh(['Test', 'Train'], err)
for i, v in enumerate(err):
    plt.text(v, i, str(np.round(v, 5)))
plt.ylabel("R^2 error measurement")
plt.suptitle("Linear Regression R^2 for Test and Train Data")
plt.show()

#######################################################################
x = df2[['weight', 'acceleration', 'horsepower']]
y = df2['mpg']
#This parameter controls the shuffling applied to the data before applying the split.
# Pass an int for reproducible output across multiple function calls.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 / 4, random_state=100)
rg = LinearRegression()

rg.fit(x_train, y_train)
err = []

plt.title("variables - 'weight', 'acceleration', 'horsepower' ")
err.append(r2_score(y_test, rg.predict(x_test)))  # test MSE
err.append(r2_score(y_train, rg.predict(x_train)))  # train MSE
plt.barh(['Test', 'Train'], err)
for i, v in enumerate(err):
    plt.text(v, i, str(np.round(v, 5)))
plt.ylabel("R^2 error measurement")
plt.suptitle("Linear Regression R^2 for Test and Train Data")
plt.show()

###############################################
x = df2[['cylinders', 'acceleration', 'displacement']]
y = df2['mpg']
#This parameter controls the shuffling applied to the data before applying the split.
# Pass an int for reproducible output across multiple function calls.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 / 4, random_state=100)
rg = LinearRegression()

rg.fit(x_train, y_train)
err = []

plt.title("variables - 'cylinders', 'acceleration', 'displacement' ")
err.append(r2_score(y_test, rg.predict(x_test)))  # test MSE
err.append(r2_score(y_train, rg.predict(x_train)))  # train MSE
plt.barh(['Test', 'Train'], err)
for i, v in enumerate(err):
    plt.text(v, i, str(np.round(v, 5)))
plt.ylabel("R^2 error measurement")
plt.suptitle("Linear Regression R^2 for Test and Train Data")
plt.show()

#########################################################################################
#######################VARIABLES TRANSFORMATION ###############################

# --> log Data
########################################################################################
x= df2[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration']]
y = df2['mpg']
#This parameter controls the shuffling applied to the data before applying the split.
# Pass an int for reproducible output across multiple function calls.
x_train, x_test, y_train, y_test = train_test_split(np.log(x), y, test_size=1 / 3, random_state=100)
mrl = LinearRegression()

rg.fit(x_train, y_train)
err = []

plt.title("All variable taken into account ")
err.append(r2_score(y_test, rg.predict(x_test)))  # test MSE
err.append(r2_score(y_train, rg.predict(x_train)))  # train MSE
plt.barh(['Test', 'Train'], err)
for i, v in enumerate(err):
    plt.text(v, i, str(np.round(v, 5)))
plt.ylabel("R^2")
plt.suptitle("Linear Regression R^2 for Test and Train Data")
plt.show()
################################################################################
x= df2[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration']]
y = df2['mpg']
#This parameter controls the shuffling applied to the data before applying the split.
# Pass an int for reproducible output across multiple function calls.
x_train, x_test, y_train, y_test = train_test_split(np.square(x), y, test_size=1 / 5, random_state=100)
mrl = LinearRegression()

rg.fit(x_train, y_train)
err = []

plt.title("All variable taken into account ")
err.append(r2_score(y_test, rg.predict(x_test)))  # test MSE
err.append(r2_score(y_train, rg.predict(x_train)))  # train MSE
plt.barh(['Test', 'Train'], err)
for i, v in enumerate(err):
    plt.text(v, i, str(np.round(v, 5)))
plt.ylabel("R^2")
plt.suptitle("Linear Regression R^2 for Test and Train Data")
plt.show()


#####################################################3
x= df2[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration']]
y = df2['mpg']
#This parameter controls the shuffling applied to the data before applying the split.
# Pass an int for reproducible output across multiple function calls.
x_train, x_test, y_train, y_test = train_test_split(np.sqrt(x), y, test_size=1 / 5, random_state=100)
mrl = LinearRegression()

rg.fit(x_train, y_train)
err = []

plt.title("All variable taken into account ")
err.append(r2_score(y_test, rg.predict(x_test)))  # test MSE
err.append(r2_score(y_train, rg.predict(x_train)))  # train MSE
plt.barh(['Test', 'Train'], err)
for i, v in enumerate(err):
    plt.text(v, i, str(np.round(v, 5)))
plt.ylabel("R^2")
plt.suptitle("Linear Regression R^2 for Test and Train Data")
plt.show()
###########################################################
x= df2[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration']]
y = df2['mpg']
#This parameter controls the shuffling applied to the data before applying the split.
# Pass an int for reproducible output across multiple function calls.
x_train, x_test, y_train, y_test = train_test_split(1/x, y, test_size=1 / 5, random_state=100)
mrl = LinearRegression()

rg.fit(x_train, y_train)
err = []

plt.title("All variable taken into account ")
err.append(r2_score(y_test, rg.predict(x_test)))  # test MSE
err.append(r2_score(y_train, rg.predict(x_train)))  # train MSE
plt.barh(['Test', 'Train'], err)
for i, v in enumerate(err):
    plt.text(v, i, str(np.round(v, 5)))
plt.ylabel("R^2")
plt.suptitle("Linear Regression R^2 for Test and Train Data")
plt.show()
