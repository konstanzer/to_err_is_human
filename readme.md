## Zestimate Errors

<img src="img/z.png" width="200"/>

---
- [Summary](#introduction)
- [Data](#data)
- [Planning pipeline](#planning-pipeline)
- [Hypotheses](#hypotheses)
- [Results](#results)
- [Recommendations](#recommendations)

## Summary

According to the 2017 Zillow competition on Kaggle, real estate company Zillow used *millions* machine learning and statistical models and hundreds of features for each property to produce home value estimates with a median error of five percent. In this project, I fail to predict drivers of the Zestimate log error using a combination of clustering algorithms and linear modeling. 

## Data

| Feature                        | Description                                                                                                            |
|--------------------------------|------------------------------------------------------------------------------------------------------------------------|
| 'architecturalstyletypeid'     |  Architectural style of the home (i.e. ranch, colonial, split-level, etc…)                                             |
| 'basementsqft'                 |  Finished living area below or partially below ground level                                                            |
| 'bathroomcnt'                  |  Number of bathrooms in home including fractional bathrooms                                                            |
| 'bedroomcnt'                   |  Number of bedrooms in home                                                                                            |
| 'buildingqualitytypeid'        |  Overall assessment of condition of the building from best (lowest) to worst (highest)                                 |                                            |
| 'calculatedfinishedsquarefeet' |  Calculated total finished living area of the home                                                                     |                                         |
| 'fips'                         |  Federal Information Processing Standard code -  see https://en.wikipedia.org/wiki/FIPS_county_code  |
| 'fireplacecnt'                 |  Number of fireplaces in a home (if any)                                                                               |
| 'garagecarcnt'                 |  Total number of garages on the lot including an attached garage                                                       |  
| 'latitude'                     |  Latitude of the middle of the parcel multiplied by 10e6                                                               |
| 'longitude'                    |  Longitude of the middle of the parcel multiplied by 10e6                                                              |
| 'lotsizesquarefeet'            |  Area of the lot in square feet                                                                                        |
| 'regionidcounty'               | County in which the property is located                                                                                |
| 'regionidcity'                 |  City in which the property is located (if any)                                                                        |
| 'regionidzip'                  |  Zip code in which the property is located                                                                             |   
| 'yearbuilt'                    |  The Year the principal residence was built                                                                            |
| 'taxvaluedollarcnt'            | The total tax assessed value of the parcel                                                                             |
| 'structuretaxvaluedollarcnt'   | The assessed value of the built structure on the parcel                                                                |
| 'landtaxvaluedollarcnt'        | The assessed value of the land area of the parcel                                                                      |
| 'taxamount'                    | The total property tax assessed for that assessment year                                                               |
The target variable is log error, which is the log of the Zestimate minus the log of the actual sale price. It's a clever way to deal with right-skewed, heteroskedastic home values because it makes errors relative to to home value. It also avoid the downfall of using percentage error which favors models with negative errors.

## Planning pipeline

Step 1: Acquire

If data is in a SQL database, run select * from zillow.2017 via SQL IDE.
If data is a csv file, use pandas, e.g. pandas.read_csv().
These steps are covered in acquire.py.

Step 3: Prepare Data

There are no missing values.
Convert all features to floats or categorical variables using one-hot encoding.
Split data into 70/15/15 training/validation/test sets.
These steps are covered in prepare.py.

Step 4: Explore & Preprocess

Visualize attributes & interactions (Python: seaborn and matplotlib).
Analyze: statistically and more generally (Python: statsmodels, numpy, scipy, scikit-learn).

Step 5: Model

Create clusters that don't predict logerror and then discard them. Use price per square foot to make a linear model that mimics the basline model, then discard those noisy predictions and settle for predicting the mean log error on every property.

Models used:

* k-means
* LinearRegression

## Hypotheses

Location (county and/or zip code) is driving log error.
Finished square feet is driving log error.
The ratio of building size to lot size is driving log error.
Clusters based on latitude and longitude will identify coastal properties an these are driving log error.
Properties with many outlying features are driving log error.

## Results

| model | RMSE | R^2
| --- | --- | --- |
| mean | .16 | 0 |
| OLS linear regression | .16 | 0 |

## Recommendations

Given the data available, I am confident in saying there are no significant drivers of the targer (log error), and that this is true for both the original features and new features derived from those. I am not surprised because the Zestimate is very accurate to begin with. I would investigate next why the model is making a few hundred huge errors in the training set that are doubling the median error. In the most extreme example, the model is predicting over $1 billion for a $6 million property (where the log error is over 5.)
