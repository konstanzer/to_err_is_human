import pandas as pd
from env import host, username, password

def get_db_url(username, host, password, db):
    return f'mysql+pymysql://{username}:{password}@{host}/{db}'

def get_zillow_data():
    url = get_db_url(username, host, password, 'zillow')
    #query to select only last transaction for given property
    query = """
            SELECT prop.parcelid
            , pred.logerror
            , bathroomcnt
            , bedroomcnt
            , poolcnt
            , buildingqualitytypeid
            , calculatedfinishedsquarefeet
            , fips
            , latitude
            , longitude
            , lotsizesquarefeet
            , regionidzip
            , yearbuilt
            , structuretaxvaluedollarcnt
            , taxvaluedollarcnt
            , landtaxvaluedollarcnt
            , taxamount
            FROM predictions_2017 pred
            JOIN properties_2017 prop USING(parcelid)
            INNER JOIN
              (SELECT parcelid, MAX(transactiondate) maxdate
               FROM predictions_2017
               GROUP BY parcelid) g
            ON pred.parcelid = g.parcelid AND pred.transactiondate = g.maxdate
            WHERE propertylandusetypeid IN (260,261,262,263,264,265,266,275);
            """
    return pd.read_sql(query, url)