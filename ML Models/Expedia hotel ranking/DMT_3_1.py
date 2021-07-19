import numpy as np
import pandas as pd
import pickle
pd.options.display.max_columns=None
import xgboost as xgb
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

avg_numerics = pickle.load(open('C:/Users/mvkri/OneDrive/Documents/DMT/data/numeric_per_prop_id_avg_mean_std_competitors.pkl','rb'))
df_test = pd.read_csv('C:/Users/mvkri/OneDrive/Documents/DMT/test_set_VU_DM.csv')

##filename=('C:/Users/mvkri/OneDrive/Documents/DMT/Xgboost_model.sav')
#loaded_model = pickle.load(open(filename, 'rb'))
preprocessing_3(df_test,type = 0)
# Add extra features to true test data

df_test['starrating_diff'] = abs(
df_test['visitor_hist_starrating'] - df_test['prop_starrating'])
# 2. usd_diff = |visitor_hist_adr_usd - price_usd|
df_test['usd_diff'] = abs(df_test['visitor_hist_adr_usd'] - df_test['price_usd'])
# df_bool1 = df_test.loc[df_test['booking_bool'] == 1]
# mean_bool1 = df_bool1['prop_starrating'].mean()
mean_bool1 = 3.3120601199508637
df_test['mean_bool'] = mean_bool1
df_test['prop_starating_monotonic'] = abs(
df_test['prop_starrating'] - df_test['mean_bool'])
df_test = df_test.drop('mean_bool', 1)
print ("Feature extraction completed")

hotel_df = pd.read_csv('C:/Users/mvkri/OneDrive/Documents/DMT/dict.csv', delimiter=',',header=None)
hotel_values = hotel_df[0].to_list()
value_values = hotel_df[1].to_list()
hotel_dict = {}
for i in range(0,len(hotel_values)):
    hotel_dict[hotel_values[i]] = {}
    values_list = value_values[i].split(',')
    hotel_dict[hotel_values[i]]['hotel_count'] = values_list[0][values_list[0].find("[")+1:values_list[0].find("]")]
    hotel_dict[hotel_values[i]]['prob_b'] = values_list[2][values_list[2].find("[")+1:values_list[2].find("]")]
    hotel_dict[hotel_values[i]]['prob_c'] = values_list[3][values_list[3].find("[")+1:values_list[3].find("]")]
df_test["prop_id"] = pd.to_numeric(df_test["prop_id"],downcast='integer')
prop_id_list = list(df_test['prop_id'])
# for hotel in df_test['prop_id'].unique():
#     hotel_int = hotel
#     if int(hotel) in hotel_dict:
#         df_test.loc[(df_test['prop_id'] == hotel), 'count_hotel'] = hotel_dict[hotel]['hotel_count']
#         df_test.loc[(df_test['prop_id'] == hotel), 'prob_book'] = hotel_dict[hotel]['prob_b']
#         df_test.loc[(df_test['prop_id'] == hotel), 'prob_click'] = hotel_dict[hotel]['prob_c']

# for hotel in df_test['prop_id'].unique():
#     df_test.loc[(df_test['prop_id'] == hotel), 'count_hotel'] = hotel_count
#     df_test.loc[(df_test['prop_id'] == hotel), 'prob_book'] = prob_b
#     df_test.loc[(df_test['prop_id'] == hotel), 'prob_click'] = prob_b
count_hotel_lst =[]
prob_b_lst = []
prob_c_lst = []
for hotels in prop_id_list:
    if hotels in hotel_dict:
        #print(hotels)
        count_hotel_lst.append(hotel_dict[hotels]['hotel_count'])
        prob_b_lst.append(hotel_dict[hotels]['prob_b'])
        prob_c_lst.append(hotel_dict[hotels]['prob_c'])
    else:
        count_hotel_lst.append(0)
        prob_b_lst.append(0)
        prob_c_lst.append(0)
df_test['count_hotel'] = count_hotel_lst
df_test['prob_book'] = prob_b_lst
df_test['prob_click'] = prob_c_lst
print(df_test.head())
test_data_new = pd.merge(df_test,avg_numerics,how='left',left_on='prop_id',right_on='prop_id_',sort=False)
#srch_id_lst = df_test['srch_id'].to_list()
#prop_id_lst = df_test['prop_id'].to_list()
#del df_test
#final_df['pred'] = pred
#test_data_new.drop(['prop_id','prop_id_'],axis=1,inplace=True)
# col_names = list(test_data_new.columns)
# col_names.remove('click_bool')
# col_names.remove('booking_bool')
#del test_data_new['srch_id']
#del test_data_new['prop_id']
#del avg_numerics
#del test_data_new['prop_id']
#print(test_data_new.head())


test_data_new.to_csv('C:/Users/mvkri/OneDrive/Documents/DMT/modified_test_data_prob.csv')
# pred = loaded_model.predict(test_data_new)
# del test_data_new
# final_df = pd.DataFrame()
# final_df['srch_id'] = srch_id_lst
# final_df['prop_id'] = prop_id_lst
# final_df['pred'] = pred
# final_df.sort_values(['srch_id', 'pred'], ascending=(True,False),inplace=True)
# final_df.to_csv('C:/Users/mvkri/OneDrive/Documents/DMT/Lambda_1_target.csv',index=False)