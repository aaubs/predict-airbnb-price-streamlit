# Load packages (comments for more special stuff)

import pandas as pd
import pickle # un-pickling stuff from training notebook
from xgboost import XGBRegressor # we use a trained XGBoost model...and therefore need to load it
from sklearn.preprocessing import StandardScaler
import shap # add prediction explainability

import numpy as np
import itertools # we need that to flatten ohe.categories_ into one list for columns
import streamlit as st
from streamlit_shap import st_shap # wrapper to display nice shap viz in the app

st.set_page_config(
    page_title="Airbnb Price Prediction 🫠",
    page_icon="💸")

st.title('Predict Airbnb Prices in CPH')

#this is how you can add images e.g. from unsplash (or loca image file)
st.image('https://source.unsplash.com/0PSCd1wIrm4', caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

# use this decorator (--> @st.experimental_singleton) and 0-parameters function to only load and preprocess once
@st.experimental_singleton
def read_objects():
    model_xgb = pickle.load(open('model_xgb.pkl','rb'))
    scaler = pickle.load(open('scaler.pkl','rb'))
    ohe = pickle.load(open('ohe.pkl','rb'))
    shap_values = pickle.load(open('shap_values.pkl','rb'))
    cats = list(itertools.chain(*ohe.categories_))
    return model_xgb, scaler, ohe, cats, shap_values

model_xgb, scaler, ohe, cats, shap_values = read_objects()

# define explainer
explainer = shap.TreeExplainer(model_xgb)

#write some markdown blah
with st.expander("What's that app?"):
    st.markdown("""
    This app will help you determine what you should be asking people to pay per night for staying at your awesome place.
    We trained an AI on successful places in Copenhagen. It will give you a pricing suggestion given a few inputs.
    We recommend going around 350kr up or down depending on the amenities that you can provide and the quality of your place.
    As a little extra 🌟, we added an AI explainer 🤖 to understand factors driving prices up or down.
    """)

st.subheader('Describe your place in numbers!')

# here you collect all inputs from the user into new objects
n_hood = st.selectbox('Select Neighbourhood', options=ohe.categories_[0])
room_type = st.radio('Select Room Type', options=ohe.categories_[1])
instant_bookable = st.checkbox('Instant Bookable')
accommodates = st.number_input('How many guest can come?', min_value=1, max_value=999)
bedrooms = st.number_input('How many bedrooms are there?', min_value=1, max_value=999)
beds = st.number_input('How many beds do you provide?', min_value=1, max_value=999)
min_nights = st.number_input('How many nights should guest stay at least?', min_value=1, max_value=999)

# make a nice button that triggers creation of a new data-line in the format that the model expects and prediction
if st.button('Predict! 🚀'):
    # make a DF for categories and transform with one-hot-encoder
    new_df_cat = pd.DataFrame({'neighbourhood_cleansed':n_hood,
                'room_type':room_type}, index=[0])
    new_values_cat = pd.DataFrame(ohe.transform(new_df_cat), columns = cats , index=[0])

    # make a DF for the numericals and standard scale
    new_df_num = pd.DataFrame({'instant_bookable':instant_bookable, 
                            'accommodates': accommodates, 
                        'bedrooms':bedrooms, 
                        'beds':beds, 
                        'minimum_nights_avg_ntm':min_nights}, index=[0])
    new_values_num = pd.DataFrame(scaler.transform(new_df_num), columns = new_df_num.columns, index=[0])  
    
    #bring all columns together
    line_to_pred = pd.concat([new_values_num, new_values_cat], axis=1)

    #run prediction for 1 new observation
    predicted_value = model_xgb.predict(line_to_pred)[0]

    #print out result to user
    st.metric(label="Predicted price", value=f'{round(predicted_value)} kr')
    
    #print SHAP explainer to user
    st.subheader(f'Wait, why {round(predicted_value)} kr? Explain, AI 🤖:')
    shap_value = explainer.shap_values(line_to_pred)
    st_shap(shap.force_plot(explainer.expected_value, shap_value, line_to_pred), height=400, width=500)
