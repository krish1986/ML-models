import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle


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

training_data = pd.read_csv('C:/Users/mvkri/OneDrive/Documents/DMT/training_set_VU_DM.csv')


# In[4]:

test_data = pd.read_csv('C:/Users/mvkri/OneDrive/Documents/DMT/test_set_VU_DM.csv')


# In[5]:

preprocessing_3(training_data)
preprocessing_3(test_data,0)


# In[6]:

combination = pd.concat([training_data.iloc[:,:-2],test_data],axis=0)


# In[7]:

combination.drop('srch_id',axis=1,inplace=True)


# In[8]:

combination_groupby = combination.groupby('prop_id',sort=True).agg([np.median, np.mean, np.std])


# In[9]:

combination_groupby_reset_index = combination_groupby.reset_index()


# In[10]:

combination_groupby_reset_index.columns = ['_'.join(col).strip() for col in combination_groupby_reset_index.columns.values]


# In[19]:

combination_groupby_reset_index.fillna(0,inplace=True)


# In[20]:

pickle_output = open('C:/Users/mvkri/OneDrive/Documents/DMT/data/numeric_per_prop_id_avg_mean_std_competitors_1.pkl','wb')
pickle.dump(combination_groupby_reset_index,pickle_output)
pickle_output.close()
