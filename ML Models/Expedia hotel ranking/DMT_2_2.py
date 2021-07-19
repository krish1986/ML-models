import numpy as np
import pandas as pd
import pickle
pd.options.display.max_columns=None
import xgboost as xgb
from catboost import CatBoost, Pool, MetricVisualizer
from catboost import CatBoostClassifier
from catboost import Pool, cv
from catboost import cv
# Read the data into memory
training_data = pd.read_csv('C:/Users/mvkri/OneDrive/Documents/DMT/training_set_VU_DM.csv')
#validation_data = pd.read_csv('C:/Users/mvkri/OneDrive/Documents/DMT/training_set_VU_DM.csv',skiprows=2500000,nrows = 1000000, header=None, names= training_data.columns)
#test_data = pd.read_csv('C:/Users/mvkri/OneDrive/Documents/DMT/training_set_VU_DM.csv',skiprows= 3500000,nrows = 1000000, header=None,names= training_data.columns)

def downsampling(df):
    click_indices = df[df.click_bool == 1].index
    random_indices = np.random.choice(click_indices, len(df.loc[df.click_bool == 1]), replace=False)
    click_sample = df.loc[random_indices]

    not_click = df[df.click_bool == 0].index
    random_indices = np.random.choice(not_click, sum(df['click_bool']), replace=False)
    not_click_sample = df.loc[random_indices]

    train_downsample_copy = pd.concat([not_click_sample, click_sample], axis=0)

    print("Percentage of not click impressions: ",
          len(train_downsample_copy[train_downsample_copy.click_bool == 0]) / len(train_downsample_copy))
    print("Percentage of click impression: ",
          len(train_downsample_copy[train_downsample_copy.click_bool == 1]) / len(train_downsample_copy))
    print("Total number of records in resampled data: ", len(train_downsample_copy))
    return train_downsample_copy


def preprocessing_3(df, type = 1):
    """
    Preprocessing data
    Here we only drop features and transform NULL values
    After preprocessing:
        training data (type = 1): contains only training and target
        test data (type  = 0): contains only training
    No return values since all operation is done in place
    """
    to_delete = [
        'date_time',
        'site_id',
        #'visitor_location_country_id',
        #'visitor_hist_adr_usd',
        'prop_country_id',
        #'prop_id',
        #'prop_brand_bool',
        #'promotion_flag',
        'srch_destination_id',
        #'random_bool',
    ]

    for i in range(1,9):
        rate = 'comp' + str(i) + '_rate'
        inv = 'comp' + str(i) + '_inv'
        #diff = 'comp' + str(i) + '_rate_percent_diff'
        to_delete.extend([rate,inv])

    #This goes for the training data
    if type == 1:
        to_delete.extend(['position','gross_bookings_usd'])

    df.drop(to_delete, axis = 1, inplace = True)
    df.fillna(-10, inplace = True)

    print ("The preprocessing task is done.")
# Fist step of data preprocessing

train_downsample = downsampling(training_data)
del training_data

preprocessing_3(train_downsample)
#preprocessing_3(validation_data)
#preprocessing_3(test_data)

# 1. starrating_diff = |visitor_hist_starrating - prop_starrating|
train_downsample['starrating_diff'] = abs(
train_downsample['visitor_hist_starrating'] - train_downsample['prop_starrating'])
# 2. usd_diff = |visitor_hist_adr_usd - price_usd|
train_downsample['usd_diff'] = abs(train_downsample['visitor_hist_adr_usd'] - train_downsample['price_usd'])
df_bool1 = train_downsample.loc[train_downsample['booking_bool'] == 1]
mean_bool1 = df_bool1['prop_starrating'].mean()
train_downsample['mean_bool'] = mean_bool1
train_downsample['prop_starating_monotonic'] = abs(train_downsample['prop_starrating'] - train_downsample['mean_bool'])
train_downsample = train_downsample.drop('mean_bool', 1)
print ("Feature extraction completed")

train_downsample['count_hotel'] = train_downsample.groupby('prop_id')['prop_id'].transform('count')

## Prob_book
train_downsample['prob_book'] = np.nan
## count how many times each hotel is booked
for hotel in train_downsample.prop_id.unique().tolist():
    list_booking = train_downsample.loc[(train_downsample['prop_id']==hotel)]['booking_bool'].value_counts().tolist()
    # Some hotels are never booked, then the list is only [6], e.g. it appears 6 times but never booked
    if len(list_booking) > 1:
        booked_hotels = list_booking[1]
        total_times = sum(list_booking)
        prob_booked = booked_hotels / float(total_times)
        train_downsample.loc[(train_downsample['prop_id']==hotel), 'prob_book'] = prob_booked
train_downsample['prob_book'] = train_downsample['prob_book'].fillna(-10)

print ("prob book added")
## Prob_click
train_downsample['prob_click'] = np.nan
## count how many times each hotel is booked
for hotel in train_downsample.prop_id.unique().tolist():
    list_clicking = train_downsample.loc[(train_downsample['prop_id']==hotel)]['click_bool'].value_counts().tolist()
    # Some hotels are never clicked, then the list is only [6], e.g. it appears 6 times but never clicked
    if len(list_clicking) > 1:
        clicked_hotels = list_clicking[1]
        total_time = sum(list_clicking)
        prob_clicked = clicked_hotels / float(total_time)
        train_downsample.loc[(train_downsample['prop_id']==hotel), 'prob_click'] = prob_clicked
train_downsample['prob_click'] = train_downsample['prob_click'].fillna(-10)

print ("prob click added")
# A relevance function to define the relevance score for NDCG
# def relevance(a):
#     if a[0] == a[1] == 1:
#         return 5
#     elif a[0] == 1 and a[1] == 0:
#         return 1
#     else:
#         return 0
#

train_downsample['relevance_score'] = 0
indices_bookings = train_downsample[train_downsample['booking_bool']==1].index.to_list()
train_downsample.loc[indices_bookings, 'relevance_score'] = 5
indices_clicks = (train_downsample[(train_downsample['booking_bool']==0) & (train_downsample['click_bool']==1)].index.tolist())
train_downsample.loc[indices_clicks, 'relevance_score'] = 1
print(train_downsample.head(20))
cutoff_id = train_downsample["srch_id"].quantile(0.94) # 90/10 split
training_data_spl = train_downsample.loc[train_downsample.srch_id< cutoff_id].drop(['relevance_score'],axis=1)
validation_data = train_downsample.loc[train_downsample.srch_id>= cutoff_id].drop(['relevance_score'],axis=1)
# y_train = training_data_spl.iloc[:,-8:2].apply(relevance,axis = 1)
# y_val = validation_data.iloc[:,-8:2].apply(relevance,axis = 1)
y_train = train_downsample.loc[train_downsample.srch_id< cutoff_id]["relevance_score"]
y_val = train_downsample.loc[train_downsample.srch_id>= cutoff_id]["relevance_score"]
#groups = training_data_spl.groupby('srch_id').size().to_frame('size')['size'].to_numpy()
#X_eval['id'] = X_eval['srch_id']

# print(y_train.head())
# Read the attribute dictionary (numeric attributes per prop_id)
avg_numerics = pickle.load(open('C:/Users/mvkri/OneDrive/Documents/DMT/data/numeric_per_prop_id_avg_mean_std_competitors.pkl','rb'))

# Add the new attributes to the original data
training_data_new = pd.merge(training_data_spl,avg_numerics,how='left',left_on='prop_id',right_on='prop_id_',sort=False)
del train_downsample
del training_data_spl
validation_data_new = pd.merge(validation_data,avg_numerics,how='left',left_on='prop_id',right_on='prop_id_',sort=False)
#test_data_new = pd.merge(test_data,avg_numerics,how='left',left_on='prop_id',right_on='prop_id_',sort=False)
training_data_new.drop(['prop_id','prop_id_'],axis=1,inplace=True)
validation_data_new.drop(['prop_id','prop_id_'],axis=1,inplace=True)
#test_data_new.drop(['prop_id','prop_id_'],axis=1,inplace=True)

# col_names = list(training_data_new.columns)
# col_names.remove('click_bool')
# col_names.remove('booking_bool')
# col_names.remove('srch_id')

#y_test = test_data.iloc[:,-2:].apply(relevance,axis = 1)
# y_train = training_data.iloc[:,-2:].apply(relevance,axis = 1)
# y_val = validation_data.iloc[:,-2:].apply(relevance,axis = 1)
#y_test = test_data.iloc[:,-2:].apply(relevance,axis = 1)

groups_train = training_data_new.groupby('srch_id').size().to_frame('size')['size'].to_numpy()
groups_val = validation_data_new.groupby('srch_id').size().to_frame('size')['size'].to_numpy()

variables_to_remove=['click_bool', 'booking_bool', 'srch_id']
training_data_new.drop(variables_to_remove,axis=1,inplace=True)
validation_data_new.drop(variables_to_remove,axis=1,inplace=True)

model = xgb.XGBRanker(n_estimators=2000, max_depth=10, n_jobs=-1, objective='rank:ndcg', learning_rate=0.1)
model.fit(training_data_new,y_train,group=groups_train,verbose=True, eval_set =[(validation_data_new,y_val)], eval_group=[groups_val])

filename=('C:/Users/mvkri/OneDrive/Documents/DMT/Xgboost_model_learning.sav')
pickle.dump(model, open(filename, 'wb'))

hotel_dict = {}
for hotel in train_downsample['prop_id'].unique():
    if hotel in hotel_dict:
        hotel_dict[hotel]['hotel_count'] = train_downsample[train_downsample['prop_id'] == hotel]['count_hotel'].unique()
        hotel_dict[hotel]['prob_b'] = train_downsample[train_downsample['prop_id'] == hotel]['prob_book'].unique()
        hotel_dict[hotel]['prob_c'] = train_downsample[train_downsample['prop_id'] == hotel]['prob_click'].unique()
    else:
        hotel_dict[hotel] = {}
        hotel_dict[hotel]['hotel_count'] = train_downsample[train_downsample['prop_id'] == hotel]['count_hotel'].unique()
        hotel_dict[hotel]['prob_b'] = train_downsample[train_downsample['prop_id'] == hotel]['prob_book'].unique()
        hotel_dict[hotel]['prob_c'] = train_downsample[train_downsample['prop_id'] == hotel]['prob_click'].unique()

import csv
a_file = open("C:/Users/mvkri/OneDrive/Documents/DMT/dict.csv", "w")

writer = csv.writer(a_file)
for key, value in hotel_dict.items():
    writer.writerow([key, value])

a_file.close()