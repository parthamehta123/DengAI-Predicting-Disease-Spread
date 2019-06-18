
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt


# In[42]:


dengue_features_df = pd.read_csv("dengue_features_train.csv")
dengue_features_df.head()


# In[43]:


new_den_fea_df = dengue_features_df.copy()


# In[44]:


new_den_fea_df.head()


# In[45]:


new_den_fea_df= new_den_fea_df.reindex(np.random.permutation(new_den_fea_df.index))


# In[46]:


dengue_features_df.head()


# In[47]:


new_den_fea_df.head()


# In[48]:


dengue_feature_labels = pd.read_csv("dengue_labels_train.csv")


# In[49]:


dengue_features_df = pd.concat([dengue_features_df, dengue_feature_labels["total_cases"]], axis=1)


# In[50]:


dengue_features_df.shape


# In[51]:


dengue_features_df.keys()


# In[53]:


dengue_features_df.isnull().sum()


# In[56]:


type(dengue_features_df["ndvi_ne"].head()[1])


# In[73]:


dengue_features_df["ndvi_ne"] = dengue_features_df["ndvi_ne"].fillna(dengue_features_df["ndvi_ne"].mean())
dengue_features_df["ndvi_nw"] = dengue_features_df["ndvi_nw"].fillna(dengue_features_df["ndvi_nw"].mean())
dengue_features_df["ndvi_se"] = dengue_features_df["ndvi_se"].fillna(dengue_features_df["ndvi_se"].mean())
dengue_features_df["ndvi_sw"] = dengue_features_df["ndvi_sw"].fillna(dengue_features_df["ndvi_sw"].mean())
dengue_features_df["precipitation_amt_mm"] = dengue_features_df["precipitation_amt_mm"].fillna(dengue_features_df["precipitation_amt_mm"].mean())
dengue_features_df["reanalysis_air_temp_k"] = dengue_features_df["reanalysis_air_temp_k"].fillna(dengue_features_df["reanalysis_air_temp_k"].mean())
dengue_features_df["reanalysis_avg_temp_k"] = dengue_features_df["reanalysis_avg_temp_k"].fillna(dengue_features_df["reanalysis_avg_temp_k"].mean())
dengue_features_df["reanalysis_dew_point_temp_k"] = dengue_features_df["reanalysis_dew_point_temp_k"].fillna(dengue_features_df["reanalysis_dew_point_temp_k"].mean())
dengue_features_df["reanalysis_max_air_temp_k"] = dengue_features_df["reanalysis_max_air_temp_k"].fillna(dengue_features_df["reanalysis_max_air_temp_k"].mean())
dengue_features_df["reanalysis_min_air_temp_k"] = dengue_features_df["reanalysis_min_air_temp_k"].fillna(dengue_features_df["reanalysis_min_air_temp_k"].mean())
dengue_features_df["reanalysis_precip_amt_kg_per_m2"] = dengue_features_df["reanalysis_precip_amt_kg_per_m2"].fillna(dengue_features_df["reanalysis_precip_amt_kg_per_m2"].mean())
dengue_features_df["reanalysis_relative_humidity_percent"] = dengue_features_df["reanalysis_relative_humidity_percent"].fillna(dengue_features_df["reanalysis_relative_humidity_percent"].mean())
dengue_features_df["reanalysis_sat_precip_amt_mm"] = dengue_features_df["reanalysis_sat_precip_amt_mm"].fillna(dengue_features_df["reanalysis_sat_precip_amt_mm"].mean())
dengue_features_df["reanalysis_specific_humidity_g_per_kg"] = dengue_features_df["reanalysis_specific_humidity_g_per_kg"].fillna(dengue_features_df["reanalysis_specific_humidity_g_per_kg"].mean())
dengue_features_df["reanalysis_tdtr_k"] = dengue_features_df["reanalysis_tdtr_k"].fillna(dengue_features_df["reanalysis_tdtr_k"].mean())
dengue_features_df["station_avg_temp_c"] = dengue_features_df["station_avg_temp_c"].fillna(dengue_features_df["station_avg_temp_c"].mean())
dengue_features_df["station_diur_temp_rng_c"] = dengue_features_df["station_diur_temp_rng_c"].fillna(dengue_features_df["station_diur_temp_rng_c"].mean())
dengue_features_df["station_max_temp_c"] = dengue_features_df["station_max_temp_c"].fillna(dengue_features_df["station_max_temp_c"].mean())
dengue_features_df["station_min_temp_c"] = dengue_features_df["station_min_temp_c"].fillna(dengue_features_df["station_min_temp_c"].mean())
dengue_features_df["station_precip_mm"] = dengue_features_df["station_precip_mm"].fillna(dengue_features_df["station_precip_mm"].mean())


# In[74]:


dengue_features_df.isnull().sum()


# In[84]:


dengue_features_df.describe().keys()


# In[78]:


dengue_features_df.head()


# In[94]:


new_den_fea_df = dengue_features_df.reindex(np.random.permutation(dengue_features_df.index))


# In[95]:


new_den_fea_df.head()


# In[97]:


new_den_fea_df.city.unique()
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
new_den_fea_df["city"] = encoder.fit_transform(new_den_fea_df["city"])


# In[101]:


sorted(new_den_fea_df.year.unique(), )


# In[102]:


new_den_fea_df.info()


# In[105]:


from datetime import datetime
seasons = [0,0,1,1,1,2,2,2,3,3,3,0]
new_den_fea_df["seasons"] = new_den_fea_df["week_start_date"].apply(lambda x: seasons[(datetime.strptime(x, "%Y-%m-%d").month-1)])


# In[106]:


new_den_fea_df.head()


# In[108]:


from sklearn import tree
new_den_fea_df = new_den_fea_df.drop("week_start_date", axis=1)


# In[110]:


new_den_fea_df.info()


# In[113]:


new_den_fea_df.city = new_den_fea_df.city.astype("float64")
new_den_fea_df.year = new_den_fea_df.year.astype("float64")
new_den_fea_df.weekofyear = new_den_fea_df.weekofyear.astype("float64")
new_den_fea_df.seasons = new_den_fea_df.seasons.astype("float64")


# In[114]:


new_den_fea_df.info()


# In[115]:


y_df_vals = new_den_fea_df["total_cases"].copy()


# In[120]:


x_df_vals = new_den_fea_df.drop("total_cases", axis=1)


# In[122]:


x_vals = x_df_vals.values
y_vals = y_df_vals.values


# In[129]:


y_vals = y_vals.reshape(y_vals.shape[0], 1)


# In[131]:


dengTreeModel = tree.DecisionTreeRegressor(criterion="mae", splitter="best", random_state=40)
dengTreeModel.fit(x_vals, y_vals)


# In[132]:


den_test_df = pd.read_csv("dengue_features_test.csv")
den_test_df.info()


# In[135]:


den_test_df["seasons"] = den_test_df["week_start_date"].apply(lambda x : seasons[datetime.strptime(x, "%Y-%m-%d").month-1])


# In[148]:


den_test_df["city"][den_test_df["city"] == "sj"] = 1
den_test_df["city"][den_test_df["city"] == "iq"] = 0


# In[152]:


den_test_df.city = den_test_df.city.astype("float64")
den_test_df.year = den_test_df.year.astype("float64")
den_test_df.weekofyear = den_test_df.weekofyear.astype("float64")
den_test_df.seasons = den_test_df.seasons.astype("float64")


# In[155]:


den_test_df = den_test_df.drop("week_start_date", axis=1)


# In[162]:


den_test_df.isnull().sum()


# In[163]:


den_test_df["ndvi_ne"] = den_test_df["ndvi_ne"].fillna(den_test_df["ndvi_ne"].mean())
den_test_df["ndvi_nw"] = den_test_df["ndvi_nw"].fillna(den_test_df["ndvi_nw"].mean())
den_test_df["ndvi_se"] = den_test_df["ndvi_se"].fillna(den_test_df["ndvi_se"].mean())
den_test_df["ndvi_sw"] = den_test_df["ndvi_sw"].fillna(den_test_df["ndvi_sw"].mean())
den_test_df["precipitation_amt_mm"] = den_test_df["precipitation_amt_mm"]                                                .fillna(den_test_df["precipitation_amt_mm"].mean())
den_test_df["reanalysis_air_temp_k"] = den_test_df["reanalysis_air_temp_k"]                                                .fillna(den_test_df["reanalysis_air_temp_k"].mean())
den_test_df["reanalysis_avg_temp_k"] = den_test_df["reanalysis_avg_temp_k"]                                                .fillna(den_test_df["reanalysis_avg_temp_k"].mean())
den_test_df["reanalysis_dew_point_temp_k"] = den_test_df["reanalysis_dew_point_temp_k"]                                                .fillna(den_test_df["reanalysis_dew_point_temp_k"].mean())
den_test_df["reanalysis_max_air_temp_k"] = den_test_df["reanalysis_max_air_temp_k"]                                                .fillna(den_test_df["reanalysis_max_air_temp_k"].mean())
den_test_df["reanalysis_min_air_temp_k"] = den_test_df["reanalysis_min_air_temp_k"]                                                .fillna(den_test_df["reanalysis_min_air_temp_k"].mean())
den_test_df["reanalysis_precip_amt_kg_per_m2"] = den_test_df["reanalysis_precip_amt_kg_per_m2"]                                                .fillna(den_test_df["reanalysis_precip_amt_kg_per_m2"].mean())
den_test_df["reanalysis_relative_humidity_percent"] = den_test_df["reanalysis_relative_humidity_percent"]                                                .fillna(den_test_df["reanalysis_relative_humidity_percent"].mean())
den_test_df["reanalysis_sat_precip_amt_mm"] = den_test_df["reanalysis_sat_precip_amt_mm"]                                                .fillna(den_test_df["reanalysis_sat_precip_amt_mm"].mean())
den_test_df["reanalysis_specific_humidity_g_per_kg"] = den_test_df["reanalysis_specific_humidity_g_per_kg"]                                                .fillna(den_test_df["reanalysis_specific_humidity_g_per_kg"].mean())
den_test_df["reanalysis_tdtr_k"] = den_test_df["reanalysis_tdtr_k"]                                                .fillna(den_test_df["reanalysis_tdtr_k"].mean())
den_test_df["station_avg_temp_c"] = den_test_df["station_avg_temp_c"]                                                .fillna(den_test_df["station_avg_temp_c"].mean())
den_test_df["station_diur_temp_rng_c"] = den_test_df["station_diur_temp_rng_c"]                                                .fillna(den_test_df["station_diur_temp_rng_c"].mean())
den_test_df["station_max_temp_c"] = den_test_df["station_max_temp_c"]                                                .fillna(den_test_df["station_max_temp_c"].mean())
den_test_df["station_min_temp_c"] = den_test_df["station_min_temp_c"]                                                .fillna(den_test_df["station_min_temp_c"].mean())
den_test_df["station_precip_mm"] = den_test_df["station_precip_mm"]                                                .fillna(den_test_df["station_precip_mm"].mean())


# In[177]:


den_test_preds = dengTreeModel.predict(x_df_vals.values)


# In[184]:


den_test_preds = den_test_preds.astype(int)
mean_absolute_error(y_vals, den_test_preds)


# In[185]:


from sklearn.metrics import accuracy_score


# In[191]:


den_test_preds = den_test_preds.reshape(den_test_preds.shape[0], 1)


# In[197]:


den_test_preds.shape


# In[204]:


from sklearn.externals.six import StringIO   
from IPython.display import Image
import pydotplus

dot_data = StringIO()
tree.export_graphviz(dengTreeModel, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
image = Image(graph.create_png())


# In[212]:


graph.write_png("den_image.png")


# In[213]:


dengSVCModel = SVC(decision_function_shape="ovr", probability=True)
dengSVCModel.fit(x_vals, y_vals)


# In[215]:


mean_absolute_error(y_vals, dengSVCModel.predict(x_vals))


# In[246]:


test_preds_svc = dengSVCModel.predict(den_test_df)


# In[217]:


test_preds_tree = dengTreeModel.predict(den_test_df)


# In[220]:


from sklearn.ensemble import RandomForestRegressor


# In[222]:


dengRFCModel = RandomForestRegressor(criterion="mae", random_state=40)
dengRFCModel.fit(x_vals, y_vals)


# In[223]:


dengRFCModel.score(x_vals, y_vals)


# In[224]:


dengTreeModel.score(x_vals, y_vals)


# In[225]:


dengSVCModel.score(x_vals, y_vals)


# In[226]:


test_preds_rfc = dengRFCModel.predict(den_test_df)


# In[227]:


dengRFCModel.feature_importances_


# In[232]:


plt.scatter(dengSVCModel.predict(x_vals), y_vals, c="red")
plt.scatter(dengTreeModel.predict(x_vals), y_vals, c="blue")
plt.scatter(dengRFCModel.predict(x_vals), y_vals, c="green")
plt.title("Model Predictions")
plt.xlabel("predicted_cases")
plt.ylabel("actual cases")


# In[234]:


ori_test_df = pd.read_csv("dengue_features_test.csv")
ori_test_df = ori_test_df.loc[:,["city", "year", "weekofyear"]]


# In[252]:


test_preds_svc_df = pd.DataFrame(data = test_preds_svc, columns=["total_cases"])
test_preds_rfc_df = pd.DataFrame(data=test_preds_rfc, columns = ["total_cases"])
test_preds_tree_df = pd.DataFrame(data=test_preds_tree, columns=["total_cases"])


# In[256]:


svc_gen_csv = pd.concat([ori_test_df, test_preds_svc_df], axis=1)
rfc_gen_csv = pd.concat([ori_test_df, test_preds_rfc_df], axis=1)
tree_gen_csv = pd.concat([ori_test_df, test_preds_tree_df], axis=1)


# In[265]:


svc_gen_csv.total_cases = svc_gen_csv.total_cases.astype(int)


# In[266]:


svc_gen_csv.head()


# In[267]:


rfc_gen_csv.total_cases = rfc_gen_csv.total_cases.astype(int)
tree_gen_csv.total_cases = tree_gen_csv.total_cases.astype(int)


# In[268]:


rfc_gen_csv.head()


# In[269]:


tree_gen_csv.head()


# In[270]:


den_test_df.head()


# In[271]:


svc_gen_csv.to_csv("dengue_challenge/svc_preds.csv", sep=",", index=False)
rfc_gen_csv.to_csv("dengue_challenge/rfc_preds.csv", sep=",", index=False)
tree_gen_csv.to_csv("dengue_challenge/tree_preds.csv", sep=",", index=False)