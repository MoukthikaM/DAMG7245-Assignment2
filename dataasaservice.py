import csv
import pandas as pd
import json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

# event_id = input('enter the event_id : ')
def filename_eventid(eventtype,eventid):
    df1=pd.read_csv('CATALOG.csv',low_memory=False)
    regex=eventtype+""+eventid
    # print(regex)
    df1=df1[df1.id.str.startswith(regex)]
    df1=df1['file_name']
    df1.drop_duplicates()
    return df1

#first function
@app.get("/getfiles/{event_id}")
def getfiles(event_id):
    filenames = filename_eventid('S', event_id)
    return filenames

######################################################################

def number_of_events(event_name):
  df_VIS = pd.read_csv('/Users/adh.arash/Documents/Bigdata/sevir/VIS_stats_master.csv', low_memory=False)
  counts=df_VIS['event'].value_counts()[event_name]
  return counts.item()

#second function
@app.get("/eventcount/{event_name}")
def eventcount(event_name):
    eventcount=number_of_events(event_name)
    return {'Event count': eventcount }

##########################################################################

def get_description(column):
    df_VIS = pd.read_csv('/Users/adh.arash/Documents/Bigdata/sevir/VIS_stats_master.csv', low_memory=False)
    return df_VIS[column].describe()

#third function
@app.get("/desc/{column}")
def desc(column):
    col_desc = get_description(column)
    return col_desc

########################################################################

from datetime import datetime as dt
def convert(date_time):
    format = '%Y-%m-%d'  # The format
    datetime_str = dt.strptime(date_time, '%Y-%m-%d')
 
    return datetime_str
#fourth function
@app.get("/count/{Starting_Date}/{Ending_Date}")
def count_of_storms(Starting_Date,Ending_Date):
  df1=pd.read_csv('CATALOG.csv',low_memory=False)
  df2=df1.loc[df1['time_utc'] > Starting_Date]
  df3=df1.loc[df1['time_utc'] < Ending_Date]
  counts1=df2['event_type'].value_counts()
  counts2 = df3['event_type'].value_counts()
  counts =counts1 + counts2
  print(counts)
  result = counts.to_json()
  parsed = json.loads(result)
  return parsed

#####################################fifth function##########################
@app.get("/location/{event_type}/{x}/{y}")
def getlocation(event_type,x,y):
#fetch coords by event type
  x=float(x)
  y=float(y)
  df_cat = pd.read_csv('CATALOG.csv', low_memory=False)
# Select Rows Based on event type
  values=[event_type]
  filter1 = df_cat.loc[df_cat["event_type"].isin(values)]
#select lat lons for that event type
  cols = [4,7,8,9,10,11,12]
  df_event_type = filter1[filter1.columns[cols]]
  df3 = df_event_type[(df_event_type['llcrnrlat'] < x) & (df_event_type['llcrnrlon'] < y) & (df_event_type['urcrnrlat'] > x) & (df_event_type['urcrnrlon'] > y)]
  if  df3.empty:
        return print('No Records , No data in accordance to the input in the dataframe')
  else :
        result = df3.to_json(orient="records")
        p = json.loads(result)
        return p


