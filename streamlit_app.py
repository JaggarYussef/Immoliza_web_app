import streamlit as st
import requests

st.title('Immo Eliza: Property Price Predictor')

# Define the input fields for the property attributes
#nbr_bedrooms = st.number_input('Number of Bedrooms', min_value=1, value=2)
total_area_sqm = st.number_input('Total Area (sqm)', value=120.0)
#surface_land_sqm = st.number_input('Surface Land (sqm)', value=100.0)
#latitude = st.number_input('Latitude', value=0.0)
#longitude = st.number_input('Longitude', value=0.0)
primary_energy_consumption_sqm = st.number_input('Primary Energy Consumption', min_value=100, value=2000)

# Additional model features
#zip_code = st.text_input('ZIP Code')
#primary_energy_consumption_sqm = st.number_input('Primary Energy Consumption (sqm)')
# fl_garden = st.checkbox('Garden')
# fl_terrace = st.checkbox('Terrace')
# fl_swimming_pool = st.checkbox('Swimming Pool')
# fl_floodzone = st.checkbox('Flood Zone')
# property_type = st.selectbox('Property Type', ['Apartment', 'House', 'Villa'])

# When the 'Predict' button is clicked
if st.button('Predict the property price'):
    # The API endpoint URL
    url = 'https://immoliza-web-app.onrender.com/predict'
    
    # The data to be sent to the API
    input_data = {
        #'nbr_bedrooms': nbr_bedrooms,
        'total_area_sqm': total_area_sqm,
        #'surface_land_sqm': surface_land_sqm,
        #'latitude': latitude,
        #'longitude': longitude,
        # 'construction_year': construction_year,
        # 'zip_code': zip_code,
        'primary_energy_consumption_sqm': primary_energy_consumption_sqm,
        # 'fl_garden': fl_garden,
        # 'fl_terrace': fl_terrace,
        # 'fl_swimming_pool': fl_swimming_pool,
        # 'fl_floodzone': fl_floodzone,
        # 'property_type': property_type
        }
    
    # POST request to the API
    
    response = requests.post(url, json=input_data)
    
    if response.status_code == 200:
        prediction = response.json()['predicted_price']
        st.success(f'Predicted Property Price: ${prediction:.2f}')
    else:
        st.error('Error in prediction')

