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

Create clusters that don't predict logerror and then discard them. Use price per square foot to make a lienar model that mimics the basline model, which is predicting the mean log error . See the noisy predictions that

Models used:

* k-means
* LinearRegression

## Hypotheses

### H<sub>0</sub> There is no linear correlation between log error and square footage.

### H<sub>1</sub> There is a linear correlation between log error and square footage. tenure.

* Pearson correlation = 0.000
* p-value             = 0.000

## Results

| model | beep | boop | bop |
| --- | --- | --- | --- |
| baseline | .1 | 1 | .1 |
| k-means | .1 | .1 | .1 |

* I recommend we do better.



### Deliverables
1. Git hub repository with analysis and work
2. Jupyter Notebook detailing analytical process and decisions
<hr style="border-top: 10px groove blue; margin-top: 1px; margin-bottom: 1px"></hr>

## Data Dictionary
| Variable     | Description                                       | Datatype |
|--------------|---------------------------------------------------|----------|
| parcelid     | unique identifier of parcels                      | int64    |
| bathroom     | number of bathrooms                               | float64  |
| bedroom      | number of bedrooms                                | float64  |
| sqft         | square footage of the building                    | float64  |
| latitude     | latitude coordinates of the building              | float64  |
| longitude    | longitude coordinates of the building             | float64  |
| lotsqft      | square footage of the lot of land of the property | float64  |
| yearbuilt    | date of construction                              | float64  |
| taxvalue     | appraised tax value of the property               | float64  |
| landtaxvalue | tax value of the land                             | float64  |
| taxamount    | Total amount of tax paid on the property          | float64  |
| <strong>logerror*</strong>    | log error rate of the Zestimate of the property   | float64  |
| landusedesc  | The type of property                              | object   |
| county       | Name of the county the property is located in     | object   |
    * : Target variable
<hr style="border-top: 10px groove blue; margin-top: 1px; margin-bottom: 1px"></hr>

## Project Planning
1. Create a Trello board for project management
2. Import and explore the dataset
4. Develop clusters for exploration
    - Use 3 different combinations of featurescale data with appropriate scaler
5. Scale data appropriately 
6. Create models for predicting zestimate 
7. Communicate results using a jupyter notebook
<hr style="border-top: 10px groove blue; margin-top: 1px; margin-bottom: 1px"></hr>

## Initial Hypotheses 
- Influences of log error: 
> - Hypothesis 1: County in which the property is located
> - Hypothesis 2: The appraised tax value of a property 
> - Hypothesis 3: The year when the property was built
> - Hypothesis 4: Square footage of the property 
<hr style="border-top: 10px groove blue; margin-top: 1px; margin-bottom: 1px"></hr>

## Instructions for Reproducability
To be able to reproduce this project you must:
1. have a wrangle_zillow.py and explore.py module
2. have a env.py file with adequate credentials to download the zillow database, or you can download it [here](https://www.kaggle.com/c/zillow-prize-1) at Kaggle.
3. Must have familiarity with and be able to use 
