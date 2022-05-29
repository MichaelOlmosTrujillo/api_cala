from google_play_scraper import app
import pandas as pd
import numpy as np
import json 

from google_play_scraper import Sort, reviews_all

mi_claro_reviews = reviews_all(
    'com.clarocolombia.miclaro',
    sleep_milliseconds=100, 
    lang='es',
    #country = 'co'
    sort = Sort.NEWEST
)
# json_data = open("json_data.json","w")
# json.dump(mi_claro_reviews, json_data, indent = 6)
# json_data.close()
df_mi_claro = pd.DataFrame(np.array(mi_claro_reviews), columns = ['review'])
print(df_mi_claro.head())
df_mi_claro = df_mi_claro.join(pd.DataFrame(df_mi_claro.pop('review').tolist()))
df_mi_claro.to_csv('df_mi_claro.csv', sep = ',')
print(df_mi_claro.head())