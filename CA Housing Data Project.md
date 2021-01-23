## CA Housing Data Project

David Douglas

This project takes a look at housing data from the 1990 Census and using visualization and regression models with GeoPlot and PySpark, can calculate what factors most into median housing prices absent of typical data such as acreage, condition, and proximity to cities/schools/parks. The census data consisted of median age of the buildings, number of rooms, number of bedrooms, population, households, median income, median house value and ocean proximity of a given block in CA, there are over 20433 valid data points in this set. What this project ended up showing that median income is the most influential aspect of median house value, followed by ocean proximity and median house age. The other catagories did have small effects on the prices but overall their effect was nearly negligible. The visual data shows how the median income and ocean proximity are often related, but there are also some outliers and thats why ocean proximity alone is not a good indicator.

There are no special instructions to run this notebook

This notebook uses pyspark and findspark to find the files necessary to import spark

The most recent version of pyspark can be installed via pip

pip install pyspark

And findspark can be installed by running

pip install findspark

The project uses geopandas to manage geographical data which can be installed with pip

pip install geopandas

Last of all it uses geoplot to help with the geographic data plotting, this can be installed with pip

pip install geoplot

However due to a lot of dependancies its easier to use

conda install geoplot -c conda-forge


```python
import findspark
findspark.init()

import pyspark
from pyspark.sql import SparkSession
import pyspark.sql as sql 
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import DecisionTreeRegressor
import pandas as pd
import numpy as np
import geopandas
import geoplot
import matplotlib.pyplot as plt
import pyspark.sql.types as types
```

Here is where the file is loaded in, it shouldnt need to be renamed if you are running it from the downloaded folder.


```python
#Name the data file here and then creating a spark instance

file = '/home/david/Downloads/california-housing-prices/housing.csv'
spark = SparkSession.builder.appName("ca_house_data").getOrCreate() 

# Here the schema is created so its parsed correctly, then the file is housing data file is read in
schema = types.StructType().add('longitude',types.DoubleType()).add('latitude',types.DoubleType()).add('housing_median_age',types.DoubleType()).add('total_rooms',types.DoubleType()).add('total_bedrooms',types.DoubleType()).add('population',types.DoubleType()).add('households',types.DoubleType()).add('median_income',types.DoubleType()).add('median_house_value',types.DoubleType()).add('ocean_proximity',types.StringType())
df = spark.read.format('csv').option("header", "true").schema(schema).load(file).na.drop()
```

Creating a geopandas dataframe allows for merging the latitude and longitude into one geography column which makes graphing easier. I included the median income and median house value column along to graph those values comparatively. 


```python
pdf = df.select('longitude','latitude','median_income','median_house_value','ocean_proximity').toPandas()
gdf = geopandas.GeoDataFrame(pdf[['median_income','median_house_value','ocean_proximity']],geometry=geopandas.points_from_xy(pdf.longitude.astype(np.float32), pdf.latitude.astype(np.float32)))
gdf.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>median_income</th>
      <th>median_house_value</th>
      <th>ocean_proximity</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8.3252</td>
      <td>452600.0</td>
      <td>NEAR BAY</td>
      <td>POINT (-122.23000 37.88000)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8.3014</td>
      <td>358500.0</td>
      <td>NEAR BAY</td>
      <td>POINT (-122.22000 37.86000)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.2574</td>
      <td>352100.0</td>
      <td>NEAR BAY</td>
      <td>POINT (-122.24000 37.85000)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.6431</td>
      <td>341300.0</td>
      <td>NEAR BAY</td>
      <td>POINT (-122.25000 37.85000)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.8462</td>
      <td>342200.0</td>
      <td>NEAR BAY</td>
      <td>POINT (-122.25000 37.85000)</td>
    </tr>
  </tbody>
</table>
</div>



In order to graph the points over the state, I've imported a shape file of the US States and selected CA. the geopandas library will plot the state, while the geoplot library plots the points pretty easily. The identity_scale function is the unnecessarly complicated way that geoplot changes the marker size. It just sets the point scaling function to return a static value. 

This shows the distribution of housing across the state and how the most expensive places tend to be in urban costal communities. But it also shows that there are a huge majority of places that are lower end of the housing cost spectrum.


```python
#select california from states map
states = geopandas.read_file('/home/david/Downloads/ca-county-boundaries/states.shp')
california = states[states['STATE_NAME'] == 'California']
#sets marker size to 2
def identity_scale(minval, maxval):
    def scalar(val):
        return 2
    return scalar

#plot CA in background and add points with hue based on median house value
ax  = california.plot(color='white',edgecolor='black')
geoplot.pointplot(gdf.geometry,scale = gdf.median_house_value,scale_func = identity_scale,ax=ax,hue = gdf.median_house_value,legend=True,legend_var='hue')
plt.title('Median House Values Across California in 1990')
plt.show()
```


![png](CA%20Housing%20Data%20Project_files/CA%20Housing%20Data%20Project_8_0.png)


To show how median income and median house values are correlated, in this map I scaled the points according to the median income. The previous graph is a bit swarmed with the number of close neighborhoods so it obscures some of the wealthier neighborhoods.


```python
#power scale helps increase the marker size of wealthier communities.
def power_scale(minval, maxval):
    def scalar(val):
        val = val + abs(minval) + 1
        return (val/4)**2
    return scalar

ax  = california.plot(color='white',edgecolor='black')
geoplot.pointplot(gdf.geometry,scale = gdf.median_income,scale_func= power_scale,ax=ax,hue = gdf.median_house_value,legend=True,legend_var='hue')
plt.title('Median House Values Across California in 1990 (Scaled by Median Income)')
plt.show()
```


![png](CA%20Housing%20Data%20Project_files/CA%20Housing%20Data%20Project_10_0.png)


Here is a Kernel Density Estimate Plot of the neighborhoods in CA. With the previous two graphs, it was tough to get a real sense of the density, both show Northern and Southern CA to be about as dense in neighborhoods. But the KDE plot shows that Los Angeles has a lot more people and neighborhoods and the Bay Area is second most dense. The rural part of CA don't show up on this graph.


```python
#density map of CA neighborhoods, silverman is the function for the KDE
ax2 = geoplot.kdeplot(gdf.geometry,shade=True,cmap='twilight',n_levels=30,bw='silverman')
geoplot.polyplot(california,ax=ax2,zorder=1)
plt.title('Density Map of CA Housing')
plt.show()
```


![png](CA%20Housing%20Data%20Project_files/CA%20Housing%20Data%20Project_12_0.png)


I graphed the ocean proximity of the points too, it gives some insight into the catagory. The <1H OCEAN was a surprisingly narrow strip of CA housing and knowing that allowed me to fine tune one of the regression feature to get better results.


```python
#Here the hue is the ocean proximity
ax  = california.plot(color='white',edgecolor='black')
geoplot.pointplot(gdf.geometry,scale = gdf.median_house_value,scale_func = identity_scale,ax=ax,hue = gdf.ocean_proximity,legend=True,legend_var='hue')
plt.title('Distance From Ocean of Houses California in 1990')
plt.show()
```


![png](CA%20Housing%20Data%20Project_files/CA%20Housing%20Data%20Project_14_0.png)

This function gives some statistics about the median house values. The prices range from 15000 to 500000 dollars. The standard deviation gives a baseline for analyzing the results of the regression. You wouldn't want to get a RSME value close to the standard deviation because that would imply the data isn't fit. But it also gives some idea of the variance in the data, but it won't tell how well a regression line can fit the data.

```python
df.select('median_house_value').describe().show()
```

    +-------+------------------+
    |summary|median_house_value|
    +-------+------------------+
    |  count|             20433|
    |   mean|206864.41315519012|
    | stddev|115435.66709858322|
    |    min|           14999.0|
    |    max|          500001.0|
    +-------+------------------+
    
    

Here is where the variables that get put into the features columns for the model. Then I split the data 70/30 into a training and testing set.


```python
#helps create features column
assembler =  VectorAssembler(inputCols=['housing_median_age','total_rooms','total_bedrooms','population','households','median_income','prox_index'], outputCol="features")
#random split random typed seed
train, test = df.randomSplit([0.70, 0.3], seed = 58313945)
```

This part of the code is responsible for giving a numeric value to the ocean_proximity variable. Using a string indexer made the model worse because it was randomly assigned, but if I correlated the values to what they actually meant close or far the model worked much better.


```python
#prox_to_val converts value into a distance indicator
def prox_to_val(proximity):
    if ((proximity == 'NEAR BAY') | (proximity == 'ISLAND') | (proximity == 'NEAR OCEAN')):
        return 0
    if (proximity == '<1H OCEAN'):
        return 1
    if (proximity == 'INLAND'):
        return 4
    else:
        return 2
#apply this function to train and test
udf_func = F.udf(prox_to_val,types.IntegerType())
train = train.withColumn('prox_index',udf_func('ocean_proximity'))
test = test.withColumn('prox_index',udf_func('ocean_proximity'))
```

Here I run linear regression on the training data, then get a summary of the results. I also print out the coefficients and the intercept of the model. This shows that the model uses Median House Value and Ocean Proximity as the most weighted variables.


```python

#linear regression fitting
linearRegression = GeneralizedLinearRegression().setMaxIter(20).setLabelCol('median_house_value')
pipeline_lr = Pipeline(stages=[assembler, linearRegression])
lrModel = pipeline_lr.fit(train.orderBy(F.rand()))
summary = lrModel.stages[1].summary
print("Coefficients: " + str(lrModel.stages[1].coefficients) + ", Intercept: " + str(lrModel.stages[1].intercept)) 
```

    Coefficients: [1176.1125764516307,-8.970825426185803,83.74142770603738,-34.5764897929259,75.34029362650931,40825.789259675475,-20348.388718474984], Intercept: 40638.086948814096
    

Here I create an evaluator to grab the metrics of the linear regression model on the test data. The RSME tends to be < 70,000 dollars, and have a mean average error of 50,000 and I get an R^2 value of around ~.63. The fact that 63% of the variation in the data can be explained by changes in the variable is somewhat decent. The RSME value actually gets worse if I take out any of the other variables in the features. So I can take this linear regression model to be somewhat reliable prediction.


```python
#evaluation of results, prints some predictions, RSME, R^2 and MAE
lr_pred = lrModel.transform(test)
print(lr_pred.select(['median_income','median_house_value','ocean_proximity','prediction']).orderBy(F.rand()).show(5))
lr_evaluator = RegressionEvaluator(labelCol="median_house_value", predictionCol="prediction", metricName="rmse")
rmse = lr_evaluator.evaluate(lr_pred)
print("RMSE Test = %g" % rmse)
lr_evaluator = RegressionEvaluator(labelCol="median_house_value", predictionCol="prediction", metricName="r2")
r2 = lr_evaluator.evaluate(lr_pred)
print("R^2 Test = %g" % r2)
lr_evaluator = RegressionEvaluator(labelCol="median_house_value", predictionCol="prediction", metricName="mae")
mae = lr_evaluator.evaluate(lr_pred)
print("Mean Absolute Error Test = %g" % mae)
```

    +-------------+------------------+---------------+------------------+
    |median_income|median_house_value|ocean_proximity|        prediction|
    +-------------+------------------+---------------+------------------+
    |       2.0577|           69000.0|         INLAND| 78631.72068440597|
    |       4.6944|          248200.0|      <1H OCEAN|242729.12397290004|
    |         2.25|           93400.0|      <1H OCEAN| 162105.5630500635|
    |       3.3235|           87800.0|         INLAND|160622.78081922582|
    |       3.4327|          147200.0|         INLAND|125846.93733181796|
    +-------------+------------------+---------------+------------------+
    only showing top 5 rows
    
    None
    RMSE Test = 69337.1
    R^2 Test = 0.633789
    Mean Absolute Error Test = 49994.7
    

Here I wanted to show the geographical distribution and house values of the close predictions (ones within the mean average error). Its good at predicting costs of coastal middle class communities as well as poorer inland communities. There arent any surprises here.


```python
#creates new spark df with dif column, and then similar to above displays the geographic data of values less than the mean averagae error
lr_results = lr_pred.withColumn('dif',F.abs(lr_pred['prediction']-lr_pred['median_house_value']))
lr_top = lr_results.filter(lr_results['dif']<mae)
pdf_top = lr_top.select('longitude','latitude','median_income','median_house_value','ocean_proximity','dif').toPandas()
gdf_top = geopandas.GeoDataFrame(pdf_top[['median_income','median_house_value','ocean_proximity','dif']],geometry=geopandas.points_from_xy(pdf_top.longitude.astype(np.float32), pdf_top.latitude.astype(np.float32)))
ax  = california.plot(color='white',edgecolor='black')
geoplot.pointplot(gdf_top.geometry,scale = gdf_top.dif,scale_func = identity_scale,ax=ax,hue = gdf_top.median_house_value,legend=True,legend_var='hue')
plt.title('Geographical Distrubution and Values of Closest Predictions')
plt.show()
```


![png](CA%20Housing%20Data%20Project_files/CA%20Housing%20Data%20Project_26_0.png)


Then looking at the data for the worst predictions shows that the model struggles with the richest communities as well as some others scattered around. One spot I predicted it would fail at are some of the northern California coastal communities, which it did miss most of those neighborhoods.


```python
#creates new spark df with dif column, and then similar to above displays the geographic data of values greater than the mean averagae error
lr_bot = lr_results.filter(lr_results['dif']>=mae)
pdf_bot = lr_bot.select('longitude','latitude','median_income','median_house_value','ocean_proximity','dif').toPandas()
gdf_bot = geopandas.GeoDataFrame(pdf_bot[['median_income','median_house_value','ocean_proximity','dif']],geometry=geopandas.points_from_xy(pdf_bot.longitude.astype(np.float32), pdf_bot.latitude.astype(np.float32)))
ax  = california.plot(color='white',edgecolor='black')
geoplot.pointplot(gdf_bot.geometry,scale = gdf_bot.dif,scale_func=identity_scale,ax=ax,hue = gdf_bot.median_house_value,legend=True,legend_var='hue')
plt.title('Geographical Distrubution and Values of Furthest Predictions')
plt.show()
```


![png](CA%20Housing%20Data%20Project_files/CA%20Housing%20Data%20Project_28_0.png)


I also decided to make a Decision Tree Regression Model to have another model to compare to the other data. Pretty similar to the linear regression code with a different class. It actually performs quite similarly however is slightly worse in all metrics. Given that the Decision Tree will output one of 32 possible values, the scores are quite reliable.


```python
#Decision tree regessor as a class,shows the same sets of results
dt = DecisionTreeRegressor(featuresCol ='features', labelCol = 'median_house_value')
pipeline_dt = Pipeline(stages=[assembler, dt])
dt_model  = pipeline_dt.fit(train)
dt_pred = dt_model.transform(test)
print(dt_pred.select(['median_income','median_house_value','ocean_proximity','prediction']).orderBy(F.rand()).show(5))
dt_evaluator = RegressionEvaluator(labelCol="median_house_value", predictionCol="prediction", metricName="rmse")
rmse = dt_evaluator.evaluate(dt_pred)
print("RMSE Test = %g" % rmse)
dt_evaluator = RegressionEvaluator(labelCol="median_house_value", predictionCol="prediction", metricName="r2")
r2 = dt_evaluator.evaluate(dt_pred)
print("R2 Test = %g" % r2)
dt_evaluator = RegressionEvaluator(labelCol="median_house_value", predictionCol="prediction", metricName="mae")
mae = dt_evaluator.evaluate(dt_pred)
print("Mean Absolute Error Test = %g" % mae)
```

    +-------------+------------------+---------------+------------------+
    |median_income|median_house_value|ocean_proximity|        prediction|
    +-------------+------------------+---------------+------------------+
    |        4.625|          228600.0|       NEAR BAY| 246421.4846743295|
    |       7.0059|          500001.0|      <1H OCEAN| 429315.8287292818|
    |       1.3676|           47800.0|         INLAND|   77607.197745013|
    |       2.8958|          245800.0|      <1H OCEAN|179685.76002766253|
    |          2.0|          175000.0|      <1H OCEAN|144518.12735042736|
    +-------------+------------------+---------------+------------------+
    only showing top 5 rows
    
    None
    RMSE Test = 70904.4
    R2 Test = 0.617047
    Mean Absolute Error Test = 51024.9
    

Debug String for Printing Out Tree


```python
dt_model.stages[-1].toDebugString
```




    'DecisionTreeRegressionModel: uid=DecisionTreeRegressor_222c1e78f0c4, depth=5, numNodes=63, numFeatures=7\n  If (feature 5 <= 5.0006)\n   If (feature 6 <= 2.5)\n    If (feature 5 <= 3.0392)\n     If (feature 5 <= 2.29335)\n      If (feature 4 <= 735.5)\n       Predict: 144518.12735042736\n      Else (feature 4 > 735.5)\n       Predict: 204376.51807228915\n     Else (feature 5 > 2.29335)\n      If (feature 0 <= 51.5)\n       Predict: 179685.76002766253\n      Else (feature 0 > 51.5)\n       Predict: 231980.62686567163\n    Else (feature 5 > 3.0392)\n     If (feature 0 <= 50.5)\n      If (feature 5 <= 4.05955)\n       Predict: 216054.76892605633\n      Else (feature 5 > 4.05955)\n       Predict: 246421.4846743295\n     Else (feature 0 > 50.5)\n      If (feature 2 <= 496.5)\n       Predict: 291996.6638297872\n      Else (feature 2 > 496.5)\n       Predict: 370903.7837837838\n   Else (feature 6 > 2.5)\n    If (feature 5 <= 3.0392)\n     If (feature 5 <= 2.29335)\n      If (feature 3 <= 365.5)\n       Predict: 104649.30281690141\n      Else (feature 3 > 365.5)\n       Predict: 77607.197745013\n     Else (feature 5 > 2.29335)\n      If (feature 3 <= 199.0)\n       Predict: 136297.08823529413\n      Else (feature 3 > 199.0)\n       Predict: 103618.27016520895\n    Else (feature 5 > 3.0392)\n     If (feature 5 <= 4.05955)\n      If (feature 0 <= 43.5)\n       Predict: 127901.6599078341\n      Else (feature 0 > 43.5)\n       Predict: 159733.33333333334\n     Else (feature 5 > 4.05955)\n      If (feature 0 <= 38.5)\n       Predict: 161660.1175337187\n      Else (feature 0 > 38.5)\n       Predict: 206344.975\n  Else (feature 5 > 5.0006)\n   If (feature 5 <= 6.8117)\n    If (feature 6 <= 2.5)\n     If (feature 0 <= 36.5)\n      If (feature 5 <= 5.8256)\n       Predict: 271588.0907003444\n      Else (feature 5 > 5.8256)\n       Predict: 320014.441322314\n     Else (feature 0 > 36.5)\n      If (feature 5 <= 5.8256)\n       Predict: 341269.8682926829\n      Else (feature 5 > 5.8256)\n       Predict: 402478.8015873016\n    Else (feature 6 > 2.5)\n     If (feature 0 <= 32.5)\n      If (feature 5 <= 6.2233)\n       Predict: 193014.38596491228\n      Else (feature 5 > 6.2233)\n       Predict: 253566.66666666666\n     Else (feature 0 > 32.5)\n      If (feature 4 <= 134.5)\n       Predict: 174685.7142857143\n      Else (feature 4 > 134.5)\n       Predict: 310947.79545454547\n   Else (feature 5 > 6.8117)\n    If (feature 5 <= 8.02685)\n     If (feature 0 <= 26.5)\n      If (feature 6 <= 2.5)\n       Predict: 358706.25\n      Else (feature 6 > 2.5)\n       Predict: 276108.8888888889\n     Else (feature 0 > 26.5)\n      If (feature 2 <= 79.5)\n       Predict: 250550.0\n      Else (feature 2 > 79.5)\n       Predict: 429315.8287292818\n    Else (feature 5 > 8.02685)\n     If (feature 6 <= 2.5)\n      If (feature 0 <= 17.5)\n       Predict: 438399.4095238095\n      Else (feature 0 > 17.5)\n       Predict: 480330.2\n     Else (feature 6 > 2.5)\n      If (feature 0 <= 29.5)\n       Predict: 336547.8695652174\n      Else (feature 0 > 29.5)\n       Predict: 438323.6153846154\n'



Taking a look at the feature importances, 5. Median Income is the most important followed by 6. Ocean Proximity and then 0. Median Age of Buildings. Others make almost a negligible impact. Given that the feature importances follow the same trend as the linear regression, gives some confidence in both models and the importance of those variable compared to others.


```python
#prints features
dt_model.stages[-1].featureImportances
```




    SparseVector(7, {0: 0.0478, 2: 0.005, 3: 0.0011, 4: 0.0053, 5: 0.7141, 6: 0.2268})



Showing a portion of the decision tree to show how much median income plays a role. This was converted manually.


```python
#displays html file
from IPython.display import HTML
HTML(filename="/home/david/Documents/decision_tree.html")
```




<!DOCTYPE html>
<html><head>
<meta http-equiv="content-type" content="text/html; charset=UTF-8">
<title>Untitled Diagram.drawio</title>
<link rel="stylesheet" href="decision_tree_files/common.css" charset="UTF-8" type="text/css"><style type="text/css">
@media print {
  * { -webkit-print-color-adjust: exact; }
  table.mxPageSelector { display: none; }
  hr.mxPageBreak { display: none; }
}
@media screen {
  table.mxPageSelector { position: fixed; right: 10px; top: 10px;font-family: Arial; font-size:10pt; border: solid 1px darkgray;background: white; border-collapse:collapse; }
  table.mxPageSelector td { border: solid 1px gray; padding:4px; }
  body.mxPage { background: gray; }
}
</style>
<style type="text/css">
@media screen {
  body > div { padding:30px;box-sizing:content-box; }
}
</style>
</head>
<body class="mxPage">
<div style="width: 900px; height: 1440px; overflow: hidden; break-inside: avoid; background: rgb(255, 255, 255) none repeat scroll 0% 0%; break-after: page;" class="geDisableMathJax" id="mxPage-1"><div style="width: 900px; height: 1440px; overflow: hidden; top: 0px; left: 0px; position: relative; touch-action: none;"><svg style="left: 0px; top: 0px; display: block; overflow: hidden; position: absolute;" width="900" height="1440"><g transformOrigin="0 0" transform="scale(0.75,0.75)translate(0,0)"><g></g><g><g style="visibility: visible;" transform="translate(0.5,0.5)"><rect x="540" y="110" width="120" height="40" rx="6" ry="6" fill="#ffe6cc" stroke="#d79b00" pointer-events="all"></rect></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 118px; height: 1px; padding-top: 130px; margin-left: 541px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 12px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">Ocean Proximity is not INLAND</div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><rect x="1079.75" y="0" width="120" height="60" rx="9" ry="9" fill="#ffe6cc" stroke="#d79b00" pointer-events="all"></rect></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 118px; height: 1px; padding-top: 30px; margin-left: 1081px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 12px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">Median income is less or equal to 4.992</div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><rect x="260" y="227.5" width="90" height="45" rx="6.75" ry="6.75" fill="#ffe6cc" stroke="#d79b00" pointer-events="all"></rect></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 88px; height: 1px; padding-top: 250px; margin-left: 261px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 12px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><font style="font-size: 10px">Median income is less or equal to 3.066</font></div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><rect x="105" y="320" width="90" height="45" rx="6.75" ry="6.75" fill="#ffe6cc" stroke="#d79b00" pointer-events="all"></rect></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 88px; height: 1px; padding-top: 343px; margin-left: 106px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 12px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><font style="font-size: 10px">Median income is less or equal to 2.316</font></div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><rect x="30" y="395" width="90" height="45" rx="6.75" ry="6.75" fill="#ffe6cc" stroke="#d79b00" pointer-events="all"></rect></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 88px; height: 1px; padding-top: 418px; margin-left: 31px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 12px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><font style="font-size: 10px">Total Bedrooms is less than or equal to 738.5 </font></div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><rect x="0" y="470" width="67.5" height="30" fill="#f8cecc" stroke="#b85450" pointer-events="all"></rect></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 66px; height: 1px; padding-top: 485px; margin-left: 1px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 12px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><div style="font-size: 10px"><font style="font-size: 10px">$144458.21</font></div></div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><rect x="75" y="470" width="67.5" height="30" fill="#f8cecc" stroke="#b85450" pointer-events="all"></rect></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 66px; height: 1px; padding-top: 485px; margin-left: 76px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 12px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><font style="font-size: 10px">$197900.01</font></div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><rect x="180" y="395" width="90" height="45" rx="6.75" ry="6.75" fill="#ffe6cc" stroke="#d79b00" pointer-events="all"></rect></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 88px; height: 1px; padding-top: 418px; margin-left: 181px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 12px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><font style="font-size: 10px">Housing Median Age is less than or equal to 51.5</font></div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><rect x="150" y="470" width="67.5" height="30" fill="#f8cecc" stroke="#b85450" pointer-events="all"></rect></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 66px; height: 1px; padding-top: 485px; margin-left: 151px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 12px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><font style="font-size: 10px">$181276.07</font></div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><rect x="225" y="470" width="67.5" height="30" fill="#f8cecc" stroke="#b85450" pointer-events="all"></rect></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 66px; height: 1px; padding-top: 485px; margin-left: 226px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 12px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><font style="font-size: 10px">$236886.29</font></div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><rect x="405" y="312.5" width="90" height="45" rx="6.75" ry="6.75" fill="#ffe6cc" stroke="#d79b00" pointer-events="all"></rect></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 88px; height: 1px; padding-top: 335px; margin-left: 406px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 12px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><font style="font-size: 10px">Housing Median Age is less than or equal to 50.5</font></div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><rect x="330" y="395" width="90" height="45" rx="6.75" ry="6.75" fill="#ffe6cc" stroke="#d79b00" pointer-events="all"></rect></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 88px; height: 1px; padding-top: 418px; margin-left: 331px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 12px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><font style="font-size: 10px">Median income is less or equal to 4.07</font></div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><rect x="300" y="470" width="67.5" height="30" fill="#f8cecc" stroke="#b85450" pointer-events="all"></rect></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 66px; height: 1px; padding-top: 485px; margin-left: 301px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 12px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><font style="font-size: 10px">$216930.86</font></div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><rect x="375" y="470" width="67.5" height="30" fill="#f8cecc" stroke="#b85450" pointer-events="all"></rect></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 66px; height: 1px; padding-top: 485px; margin-left: 376px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 12px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><font style="font-size: 10px">$246380.56</font></div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><rect x="472.5" y="395" width="90" height="45" rx="6.75" ry="6.75" fill="#ffe6cc" stroke="#d79b00" pointer-events="all"></rect></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 88px; height: 1px; padding-top: 418px; margin-left: 474px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 12px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><font style="font-size: 10px">Total Bedrooms is less than or equal to 476.5 </font></div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><rect x="449.25" y="470" width="67.5" height="30" fill="#f8cecc" stroke="#b85450" pointer-events="all"></rect></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 66px; height: 1px; padding-top: 485px; margin-left: 450px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 12px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><font style="font-size: 10px">$291627.21</font></div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><rect x="524.25" y="470" width="67.5" height="30" fill="#f8cecc" stroke="#b85450" pointer-events="all"></rect></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 66px; height: 1px; padding-top: 485px; margin-left: 525px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 12px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><font style="font-size: 10px">$360158.59</font></div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><rect x="868" y="227.5" width="90" height="45" rx="6.75" ry="6.75" fill="#ffe6cc" stroke="#d79b00" pointer-events="all"></rect></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 88px; height: 1px; padding-top: 250px; margin-left: 869px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 12px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><font style="font-size: 10px">Median income is less or equal to 3.066</font></div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><rect x="713" y="320" width="90" height="45" rx="6.75" ry="6.75" fill="#ffe6cc" stroke="#d79b00" pointer-events="all"></rect></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 88px; height: 1px; padding-top: 343px; margin-left: 714px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 12px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><font style="font-size: 10px">Median income is less or equal to 2.182</font></div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><rect x="638" y="395" width="90" height="45" rx="6.75" ry="6.75" fill="#ffe6cc" stroke="#d79b00" pointer-events="all"></rect></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 88px; height: 1px; padding-top: 418px; margin-left: 639px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 12px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><font style="font-size: 10px">Housing Median Age is less than or equal to 9.5<br></font></div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><rect x="608" y="470" width="67.5" height="30" fill="#f8cecc" stroke="#b85450" pointer-events="all"></rect></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 66px; height: 1px; padding-top: 485px; margin-left: 609px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 12px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><font style="font-size: 10px">$118939.39</font></div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><rect x="683" y="470" width="67.5" height="30" fill="#f8cecc" stroke="#b85450" pointer-events="all"></rect></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 66px; height: 1px; padding-top: 485px; margin-left: 684px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 12px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><font size="1">$77360.96</font></div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><rect x="788" y="395" width="90" height="45" rx="6.75" ry="6.75" fill="#ffe6cc" stroke="#d79b00" pointer-events="all"></rect></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 88px; height: 1px; padding-top: 418px; margin-left: 789px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 12px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><font style="font-size: 10px">Median income is less or equal to 2.57</font></div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><rect x="758" y="470" width="67.5" height="30" fill="#f8cecc" stroke="#b85450" pointer-events="all"></rect></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 66px; height: 1px; padding-top: 485px; margin-left: 759px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 12px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><font style="font-size: 10px">$95611.43</font></div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><rect x="833" y="470" width="67.5" height="30" fill="#f8cecc" stroke="#b85450" pointer-events="all"></rect></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 66px; height: 1px; padding-top: 485px; margin-left: 834px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 12px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><font size="1">$107400.45</font></div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><rect x="1013" y="312.5" width="90" height="45" rx="6.75" ry="6.75" fill="#ffe6cc" stroke="#d79b00" pointer-events="all"></rect></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 88px; height: 1px; padding-top: 335px; margin-left: 1014px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 12px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><font style="font-size: 10px">Median income is less or equal to 4.07<br></font></div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><rect x="938" y="395" width="90" height="45" rx="6.75" ry="6.75" fill="#ffe6cc" stroke="#d79b00" pointer-events="all"></rect></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 88px; height: 1px; padding-top: 418px; margin-left: 939px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 12px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><font style="font-size: 10px">Median income is less or equal to 3.52<br></font></div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><rect x="908" y="470" width="67.5" height="30" fill="#f8cecc" stroke="#b85450" pointer-events="all"></rect></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 66px; height: 1px; padding-top: 485px; margin-left: 909px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 12px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><font size="1">$123792.51</font></div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><rect x="983" y="470" width="67.5" height="30" fill="#f8cecc" stroke="#b85450" pointer-events="all"></rect></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 66px; height: 1px; padding-top: 485px; margin-left: 984px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 12px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><font size="1">$137066.07</font></div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><rect x="1080.5" y="395" width="90" height="45" rx="6.75" ry="6.75" fill="#ffe6cc" stroke="#d79b00" pointer-events="all"></rect></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 88px; height: 1px; padding-top: 418px; margin-left: 1082px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 12px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><font style="font-size: 10px">Housing Median Age is less than or equal to 30.5<br></font></div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><rect x="1057.25" y="470" width="67.5" height="30" fill="#f8cecc" stroke="#b85450" pointer-events="all"></rect></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 66px; height: 1px; padding-top: 485px; margin-left: 1058px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 12px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><font size="1">$159156.89</font></div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><rect x="1132.25" y="470" width="67.5" height="30" fill="#f8cecc" stroke="#b85450" pointer-events="all"></rect></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 66px; height: 1px; padding-top: 485px; margin-left: 1133px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 12px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><font style="font-size: 10px">$360158.59</font></div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><path d="M 75 440 L 38.9 466.25" fill="none" stroke="white" stroke-miterlimit="10" pointer-events="stroke" visibility="hidden" stroke-width="9"></path><path d="M 75 440 L 38.9 466.25" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="stroke"></path><path d="M 34.65 469.34 L 38.26 462.39 L 38.9 466.25 L 42.37 468.06 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="all"></path></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 1px; height: 1px; padding-top: 455px; margin-left: 54px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 11px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; background-color: #ffffff; white-space: nowrap; "><div>Yes</div></div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><path d="M 75 440 L 103.99 465.77" fill="none" stroke="white" stroke-miterlimit="10" pointer-events="stroke" visibility="hidden" stroke-width="9"></path><path d="M 75 440 L 103.99 465.77" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="stroke"></path><path d="M 107.91 469.26 L 100.36 467.22 L 103.99 465.77 L 105.01 461.99 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="all"></path></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 1px; height: 1px; padding-top: 455px; margin-left: 92px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 11px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; background-color: #ffffff; white-space: nowrap; ">No</div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><path d="M 225 440 L 188.9 466.25" fill="none" stroke="white" stroke-miterlimit="10" pointer-events="stroke" visibility="hidden" stroke-width="9"></path><path d="M 225 440 L 188.9 466.25" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="stroke"></path><path d="M 184.65 469.34 L 188.26 462.39 L 188.9 466.25 L 192.37 468.06 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="all"></path></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 1px; height: 1px; padding-top: 455px; margin-left: 204px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 11px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; background-color: #ffffff; white-space: nowrap; "><div>Yes</div></div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><path d="M 375 440 L 338.9 466.25" fill="none" stroke="white" stroke-miterlimit="10" pointer-events="stroke" visibility="hidden" stroke-width="9"></path><path d="M 375 440 L 338.9 466.25" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="stroke"></path><path d="M 334.65 469.34 L 338.26 462.39 L 338.9 466.25 L 342.37 468.06 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="all"></path></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 1px; height: 1px; padding-top: 455px; margin-left: 354px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 11px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; background-color: #ffffff; white-space: nowrap; "><div>Yes</div></div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><path d="M 517.5 440 L 487.81 465.82" fill="none" stroke="white" stroke-miterlimit="10" pointer-events="stroke" visibility="hidden" stroke-width="9"></path><path d="M 517.5 440 L 487.81 465.82" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="stroke"></path><path d="M 483.84 469.27 L 486.83 462.03 L 487.81 465.82 L 491.42 467.31 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="all"></path></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 1px; height: 1px; padding-top: 455px; margin-left: 500px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 11px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; background-color: #ffffff; white-space: nowrap; "><div>Yes</div></div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><path d="M 225 440 L 253.99 465.77" fill="none" stroke="white" stroke-miterlimit="10" pointer-events="stroke" visibility="hidden" stroke-width="9"></path><path d="M 225 440 L 253.99 465.77" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="stroke"></path><path d="M 257.91 469.26 L 250.36 467.22 L 253.99 465.77 L 255.01 461.99 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="all"></path></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 1px; height: 1px; padding-top: 455px; margin-left: 242px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 11px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; background-color: #ffffff; white-space: nowrap; ">No</div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><path d="M 517.5 440 L 552.88 466.21" fill="none" stroke="white" stroke-miterlimit="10" pointer-events="stroke" visibility="hidden" stroke-width="9"></path><path d="M 517.5 440 L 552.88 466.21" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="stroke"></path><path d="M 557.1 469.33 L 549.39 467.98 L 552.88 466.21 L 553.56 462.36 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="all"></path></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 1px; height: 1px; padding-top: 455px; margin-left: 538px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 11px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; background-color: #ffffff; white-space: nowrap; ">No</div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><path d="M 375 440 L 403.99 465.77" fill="none" stroke="white" stroke-miterlimit="10" pointer-events="stroke" visibility="hidden" stroke-width="9"></path><path d="M 375 440 L 403.99 465.77" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="stroke"></path><path d="M 407.91 469.26 L 400.36 467.22 L 403.99 465.77 L 405.01 461.99 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="all"></path></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 1px; height: 1px; padding-top: 455px; margin-left: 392px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 11px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; background-color: #ffffff; white-space: nowrap; ">No</div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><path d="M 683 440 L 646.9 466.25" fill="none" stroke="white" stroke-miterlimit="10" pointer-events="stroke" visibility="hidden" stroke-width="9"></path><path d="M 683 440 L 646.9 466.25" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="stroke"></path><path d="M 642.65 469.34 L 646.26 462.39 L 646.9 466.25 L 650.37 468.06 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="all"></path></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 1px; height: 1px; padding-top: 455px; margin-left: 662px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 11px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; background-color: #ffffff; white-space: nowrap; "><div>Yes</div></div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><path d="M 833 440 L 796.9 466.25" fill="none" stroke="white" stroke-miterlimit="10" pointer-events="stroke" visibility="hidden" stroke-width="9"></path><path d="M 833 440 L 796.9 466.25" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="stroke"></path><path d="M 792.65 469.34 L 796.26 462.39 L 796.9 466.25 L 800.37 468.06 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="all"></path></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 1px; height: 1px; padding-top: 455px; margin-left: 812px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 11px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; background-color: #ffffff; white-space: nowrap; "><div>Yes</div></div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><path d="M 983 440 L 946.9 466.25" fill="none" stroke="white" stroke-miterlimit="10" pointer-events="stroke" visibility="hidden" stroke-width="9"></path><path d="M 983 440 L 946.9 466.25" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="stroke"></path><path d="M 942.65 469.34 L 946.26 462.39 L 946.9 466.25 L 950.37 468.06 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="all"></path></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 1px; height: 1px; padding-top: 455px; margin-left: 962px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 11px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; background-color: #ffffff; white-space: nowrap; "><div>Yes</div></div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><path d="M 1125.5 440 L 1095.81 465.82" fill="none" stroke="white" stroke-miterlimit="10" pointer-events="stroke" visibility="hidden" stroke-width="9"></path><path d="M 1125.5 440 L 1095.81 465.82" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="stroke"></path><path d="M 1091.84 469.27 L 1094.83 462.03 L 1095.81 465.82 L 1099.42 467.31 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="all"></path></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 1px; height: 1px; padding-top: 455px; margin-left: 1108px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 11px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; background-color: #ffffff; white-space: nowrap; "><div>Yes</div></div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><path d="M 683 440 L 711.99 465.77" fill="none" stroke="white" stroke-miterlimit="10" pointer-events="stroke" visibility="hidden" stroke-width="9"></path><path d="M 683 440 L 711.99 465.77" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="stroke"></path><path d="M 715.91 469.26 L 708.36 467.22 L 711.99 465.77 L 713.01 461.99 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="all"></path></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 1px; height: 1px; padding-top: 455px; margin-left: 700px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 11px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; background-color: #ffffff; white-space: nowrap; ">No</div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><path d="M 833 440 L 861.99 465.77" fill="none" stroke="white" stroke-miterlimit="10" pointer-events="stroke" visibility="hidden" stroke-width="9"></path><path d="M 833 440 L 861.99 465.77" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="stroke"></path><path d="M 865.91 469.26 L 858.36 467.22 L 861.99 465.77 L 863.01 461.99 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="all"></path></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 1px; height: 1px; padding-top: 455px; margin-left: 850px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 11px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; background-color: #ffffff; white-space: nowrap; ">No</div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><path d="M 983 440 L 1011.99 465.77" fill="none" stroke="white" stroke-miterlimit="10" pointer-events="stroke" visibility="hidden" stroke-width="9"></path><path d="M 983 440 L 1011.99 465.77" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="stroke"></path><path d="M 1015.91 469.26 L 1008.36 467.22 L 1011.99 465.77 L 1013.01 461.99 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="all"></path></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 1px; height: 1px; padding-top: 455px; margin-left: 1000px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 11px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; background-color: #ffffff; white-space: nowrap; ">No</div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><path d="M 1130 440 L 1161.11 465.92" fill="none" stroke="white" stroke-miterlimit="10" pointer-events="stroke" visibility="hidden" stroke-width="9"></path><path d="M 1130 440 L 1161.11 465.92" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="stroke"></path><path d="M 1165.14 469.28 L 1157.52 467.49 L 1161.11 465.92 L 1162 462.11 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="all"></path></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 1px; height: 1px; padding-top: 455px; margin-left: 1148px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 11px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; background-color: #ffffff; white-space: nowrap; ">No</div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><path d="M 150 365 L 80.91 392.63" fill="none" stroke="white" stroke-miterlimit="10" pointer-events="stroke" visibility="hidden" stroke-width="9"></path><path d="M 150 365 L 80.91 392.63" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="stroke"></path><path d="M 76.04 394.58 L 81.24 388.74 L 80.91 392.63 L 83.84 395.23 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="all"></path></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 1px; height: 1px; padding-top: 380px; margin-left: 113px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 11px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; background-color: #ffffff; white-space: nowrap; "><div>Yes</div></div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><path d="M 450 357.5 L 380.7 392.15" fill="none" stroke="white" stroke-miterlimit="10" pointer-events="stroke" visibility="hidden" stroke-width="9"></path><path d="M 450 357.5 L 380.7 392.15" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="stroke"></path><path d="M 376 394.5 L 380.7 388.24 L 380.7 392.15 L 383.83 394.5 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="all"></path></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 1px; height: 1px; padding-top: 376px; margin-left: 412px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 11px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; background-color: #ffffff; white-space: nowrap; "><div>Yes</div></div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><path d="M 450 357.5 L 511.93 391.91" fill="none" stroke="white" stroke-miterlimit="10" pointer-events="stroke" visibility="hidden" stroke-width="9"></path><path d="M 450 357.5 L 511.93 391.91" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="stroke"></path><path d="M 516.52 394.46 L 508.7 394.12 L 511.93 391.91 L 512.1 388 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="all"></path></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 1px; height: 1px; padding-top: 376px; margin-left: 484px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 11px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; background-color: #ffffff; white-space: nowrap; ">No</div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><path d="M 150 365 L 219.09 392.63" fill="none" stroke="white" stroke-miterlimit="10" pointer-events="stroke" visibility="hidden" stroke-width="9"></path><path d="M 150 365 L 219.09 392.63" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="stroke"></path><path d="M 223.96 394.58 L 216.16 395.23 L 219.09 392.63 L 218.76 388.74 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="all"></path></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 1px; height: 1px; padding-top: 380px; margin-left: 187px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 11px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; background-color: #ffffff; white-space: nowrap; ">No</div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><path d="M 758 365 L 688.91 392.63" fill="none" stroke="white" stroke-miterlimit="10" pointer-events="stroke" visibility="hidden" stroke-width="9"></path><path d="M 758 365 L 688.91 392.63" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="stroke"></path><path d="M 684.04 394.58 L 689.24 388.74 L 688.91 392.63 L 691.84 395.23 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="all"></path></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 1px; height: 1px; padding-top: 380px; margin-left: 721px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 11px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; background-color: #ffffff; white-space: nowrap; "><div>Yes</div></div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><path d="M 1058 357.5 L 988.7 392.15" fill="none" stroke="white" stroke-miterlimit="10" pointer-events="stroke" visibility="hidden" stroke-width="9"></path><path d="M 1058 357.5 L 988.7 392.15" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="stroke"></path><path d="M 984 394.5 L 988.7 388.24 L 988.7 392.15 L 991.83 394.5 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="all"></path></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 1px; height: 1px; padding-top: 376px; margin-left: 1020px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 11px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; background-color: #ffffff; white-space: nowrap; "><div>Yes</div></div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><path d="M 758 365 L 827.09 392.63" fill="none" stroke="white" stroke-miterlimit="10" pointer-events="stroke" visibility="hidden" stroke-width="9"></path><path d="M 758 365 L 827.09 392.63" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="stroke"></path><path d="M 831.96 394.58 L 824.16 395.23 L 827.09 392.63 L 826.76 388.74 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="all"></path></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 1px; height: 1px; padding-top: 380px; margin-left: 795px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 11px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; background-color: #ffffff; white-space: nowrap; ">No</div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><path d="M 1058 357.5 L 1119.93 391.91" fill="none" stroke="white" stroke-miterlimit="10" pointer-events="stroke" visibility="hidden" stroke-width="9"></path><path d="M 1058 357.5 L 1119.93 391.91" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="stroke"></path><path d="M 1124.52 394.46 L 1116.7 394.12 L 1119.93 391.91 L 1120.1 388 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="all"></path></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 1px; height: 1px; padding-top: 376px; margin-left: 1092px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 11px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; background-color: #ffffff; white-space: nowrap; ">No</div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><path d="M 305 272.5 L 156.09 318.13" fill="none" stroke="white" stroke-miterlimit="10" pointer-events="stroke" visibility="hidden" stroke-width="9"></path><path d="M 305 272.5 L 156.09 318.13" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="stroke"></path><path d="M 151.07 319.67 L 156.74 314.28 L 156.09 318.13 L 158.79 320.97 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="all"></path></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 1px; height: 1px; padding-top: 296px; margin-left: 228px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 11px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; background-color: #ffffff; white-space: nowrap; "><div>Yes</div></div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><path d="M 305 272.5 L 443.86 310.81" fill="none" stroke="white" stroke-miterlimit="10" pointer-events="stroke" visibility="hidden" stroke-width="9"></path><path d="M 305 272.5 L 443.86 310.81" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="stroke"></path><path d="M 448.92 312.2 L 441.24 313.72 L 443.86 310.81 L 443.11 306.97 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="all"></path></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 1px; height: 1px; padding-top: 292px; margin-left: 377px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 11px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; background-color: #ffffff; white-space: nowrap; ">No</div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><path d="M 913 272.5 L 764.09 318.13" fill="none" stroke="white" stroke-miterlimit="10" pointer-events="stroke" visibility="hidden" stroke-width="9"></path><path d="M 913 272.5 L 764.09 318.13" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="stroke"></path><path d="M 759.07 319.67 L 764.74 314.28 L 764.09 318.13 L 766.79 320.97 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="all"></path></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 1px; height: 1px; padding-top: 296px; margin-left: 836px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 11px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; background-color: #ffffff; white-space: nowrap; "><div>Yes</div></div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><path d="M 600 150 L 311.16 225.88" fill="none" stroke="white" stroke-miterlimit="10" pointer-events="stroke" visibility="hidden" stroke-width="9"></path><path d="M 600 150 L 311.16 225.88" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="stroke"></path><path d="M 306.08 227.22 L 311.96 222.05 L 311.16 225.88 L 313.74 228.82 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="all"></path></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 1px; height: 1px; padding-top: 189px; margin-left: 452px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 11px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; background-color: #ffffff; white-space: nowrap; "><div>Yes</div></div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><path d="M 600 150 L 906.82 225.97" fill="none" stroke="white" stroke-miterlimit="10" pointer-events="stroke" visibility="hidden" stroke-width="9"></path><path d="M 600 150 L 906.82 225.97" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="stroke"></path><path d="M 911.91 227.23 L 904.28 228.95 L 906.82 225.97 L 905.96 222.15 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="all"></path></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 1px; height: 1px; padding-top: 189px; margin-left: 756px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 11px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; background-color: #ffffff; white-space: nowrap; ">No</div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><path d="M 913 272.5 L 1051.86 310.81" fill="none" stroke="white" stroke-miterlimit="10" pointer-events="stroke" visibility="hidden" stroke-width="9"></path><path d="M 913 272.5 L 1051.86 310.81" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="stroke"></path><path d="M 1056.92 312.2 L 1049.24 313.72 L 1051.86 310.81 L 1051.11 306.97 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="all"></path></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 1px; height: 1px; padding-top: 292px; margin-left: 985px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 11px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; background-color: #ffffff; white-space: nowrap; ">No</div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><path d="M 1139.75 60 L 606.34 109.41" fill="none" stroke="white" stroke-miterlimit="10" pointer-events="stroke" visibility="hidden" stroke-width="9"></path><path d="M 1139.75 60 L 606.34 109.41" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="stroke"></path><path d="M 601.11 109.9 L 607.76 105.77 L 606.34 109.41 L 608.41 112.74 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="all"></path></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 1px; height: 1px; padding-top: 85px; margin-left: 870px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 11px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; background-color: #ffffff; white-space: nowrap; "><div>Yes</div></div></div></div></foreignObject></g></g></g><g></g></g></svg></div></div><hr class="mxPageBreak"><div style="width: 900px; height: 1440px; overflow: hidden; break-inside: avoid; background: rgb(255, 255, 255) none repeat scroll 0% 0%;" class="geDisableMathJax" id="mxPage-2"><div style="width: 900px; height: 1440px; overflow: hidden; top: 0px; left: 0px; position: relative; touch-action: none;"><svg style="left: 0px; top: 0px; display: block; overflow: hidden; position: absolute;" width="900" height="1440"><g transformOrigin="0 0" transform="scale(0.75,0.75)translate(-1200,0)"><g></g><g><g style="visibility: visible;" transform="translate(0.5,0.5)"><rect x="1079.75" y="0" width="120" height="60" rx="9" ry="9" fill="#ffe6cc" stroke="#d79b00" pointer-events="all"></rect></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 118px; height: 1px; padding-top: 30px; margin-left: 1081px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 12px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">Median income is less or equal to 4.992</div></div></div></foreignObject></g></g><g style="visibility: visible;" transform="translate(0.5,0.5)"><rect x="1132.25" y="470" width="67.5" height="30" fill="#f8cecc" stroke="#b85450" pointer-events="all"></rect></g><g style=""><g><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%"><div style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 66px; height: 1px; padding-top: 485px; margin-left: 1133px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 12px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><font style="font-size: 10px">$360158.59</font></div></div></div></foreignObject></g></g></g><g></g></g></svg></div></div>

</body></html>




```python

```
