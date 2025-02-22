import streamlit as st
import pandas as pd
import numpy as np
import pickle
import category_encoders as ce

# Load the price scaler
with open("price_scaling.pkl", "rb") as f:
    price_scaler = pickle.load(f)


with open('be_page2.pkl', 'rb') as be:
    be_page2 = pickle.load(be)

# Load classification model
with open('random_forest.pkl', 'rb') as f:
    rf = pickle.load(f)

with open('page_label_encode.pkl','rb') as pg:
    page_label_encode = pickle.load(pg)

with open('gradient_boosting.pkl','rb') as gb:
    regressor = pickle.load(gb)    

with open('k-means.pkl','rb') as k:
    kmeans = pickle.load(k)

df_cluster = pd.read_csv('test_data.csv')
unique_clothing_models = df_cluster["page2_clothing_model"].unique().tolist()

# Streamlit UI
st.set_page_config(page_title="Price Prediction App", layout="wide")

# Navigation
page = st.sidebar.radio("Navigation", ["Welcome", "Predict Buy/Not Buy",'Price prediction','Customer Cluster'])

if page == "Welcome":
    st.title("Welcome to Price Prediction App")
    st.subheader('Customer behaviour')
    st.subheader('Price prediction')
    st.subheader('Customers Cluster segmentation based on browsing behaviours')


elif page == "Predict Buy/Not Buy":
    st.title("Classification Model: Predict Buy or Not Buy")

   
    page1_main_category = st.selectbox("Main Category", [1, 2, 3, 4])
    page1_main_category_vector = [1 if page1_main_category == i else 0 for i in [1, 2, 3, 4]]

   
    page2_cm = st.selectbox("Clothing Model", unique_clothing_models)
    page2_cm_encoded = be_page2.transform(pd.DataFrame({'page2_clothing_model': [page2_cm]}))

    page2_cm_encoded = page2_cm_encoded.values.flatten().tolist() 

    # Enter Price and Scale It
    price_input = st.number_input("Enter Price", value=0.0)
    price_scaled = price_scaler.transform([[price_input]])[0][0]  # Keep it a single value

    # --- Prepare input data ---
    input_data = pd.DataFrame([  
        page2_cm_encoded +  # Binary-encoded clothing model (8 columns)
        [price_scaled] +  # Scaled Price (1 column)
        page1_main_category_vector  # One-hot encoded main category (4 columns)
    ], columns=[
        'page2_clothing_model_0', 'page2_clothing_model_1', 
        'page2_clothing_model_2', 'page2_clothing_model_3',
        'page2_clothing_model_4', 'page2_clothing_model_5',
        'page2_clothing_model_6', 'page2_clothing_model_7',
        'price', 
        'page1_main_category_1', 'page1_main_category_2', 
        'page1_main_category_3', 'page1_main_category_4'
    ])

 

    # --- Predict ---
    if st.button("Predict"):
        prediction = rf.predict(input_data)[0]
        result = "Buy" if prediction == 1 else "Not Buy"
        st.success(f"Prediction: The customer will {result}")

elif page == "Price prediction":
    st.title('Price prediction of products')

    page1_main_category = st.selectbox("Main Category", ['Trousers','Skirts','Blouses','Sales'])
    if page1_main_category == 'Trousers':
        page1_main_category_vector =[1,0,0,0]
    elif page1_main_category == 'Skirts':
        page1_main_category_vector = [0,1,0,0]
    elif  page1_main_category == 'Blouses':
        page1_main_category_vector = [0,0,1,0]
    else:
        page1_main_category_vector = [0,0,0,1]    
    

   
    page2_cm = st.selectbox("Clothing Model", unique_clothing_models)
    page2_cm_encoded = be_page2.transform(pd.DataFrame({'page2_clothing_model': [page2_cm]}))

    page2_cm_encoded = page2_cm_encoded.values.flatten().tolist() 

    page_number = st.selectbox('select page number',[1,2,3,4,5])
    page_number_encode = [page_label_encode.transform([page_number])[0]]

    model_photography = st.selectbox('select product DP',['Enface','Profile'])
    
    model_photo_vector = [1,0] if model_photography == 'enface' else[0,1]

    price_2 = st.selectbox('whether want to buy',['Yes','No'])
    price_2_vector = [1, 0] if price_2 == "Buy" else [0, 1]

    input_dataframe = pd.DataFrame([
        page2_cm_encoded + 
        page_number_encode +
        price_2_vector + 
        model_photo_vector + 
        page1_main_category_vector], 
        columns = ['page2_clothing_model_0', 'page2_clothing_model_1',
       'page2_clothing_model_2', 'page2_clothing_model_3',
       'page2_clothing_model_4', 'page2_clothing_model_5',
       'page2_clothing_model_6', 'page2_clothing_model_7', 'page', 
       'price_2_1','price_2_2', 'model_photography_1', 'model_photography_2',
       'page1_main_category_1', 'page1_main_category_2',
       'page1_main_category_3', 'page1_main_category_4'
    ])

    if st.button("Predict"):
        price_prediction = regressor.predict(input_dataframe)[0]
        output = price_scaler.inverse_transform(np.array(price_prediction).reshape(-1, 1))[0][0]
       
        st.success(f"The expected product price is {output:.2f}")

else:
    if page == "Customer Cluster":
        st.title('Customers clusters')
        page1_main_category = st.selectbox("Main Category", [1, 2, 3, 4])
        page1_main_category_vector = [1 if page1_main_category == i else 0 for i in [1, 2, 3, 4]]

    
        page2_cm = st.selectbox("Clothing Model", unique_clothing_models)
        page2_cm_encoded = be_page2.transform(pd.DataFrame({'page2_clothing_model': [page2_cm]}))

        page2_cm_encoded = page2_cm_encoded.values.flatten().tolist() 

        # Enter Price and Scale It
        price_input = st.number_input("Enter Price", value=0.0)
        price_scaled = price_scaler.transform([[price_input]])[0][0]
        price_scaled = [price_scaled] 

        page_number = st.selectbox('select page number',[1,2,3,4,5])
        page_number_encode = [page_label_encode.transform([page_number])[0]]

        model_photography = st.selectbox('select product DP',['Enface','Profile'])
        
        model_photo_vector = [1,0] if model_photography == 'enface' else[0,1]

        price_2 = st.selectbox('whether want to buy',['Yes','No'])
        price_2_vector = [1, 0] if price_2 == "Buy" else [0, 1]
        
        color_mapping = {
        'Beige': 1, 'Black': 2, 'Blue': 3, 'Brown': 4, 'Burgundy': 5,
        'Gray': 6, 'Green': 7, 'Navy blue': 8, 'of many colors': 9,
        'Olive': 10, 'Pink': 11, 'Red': 12, 'Violet': 13, 'White': 14}

        colour = st.selectbox('Select the colour', list(color_mapping.keys()))

        colour_encoded = [color_mapping[colour]]
        
        location_options = ['top left','top in the middle','top right','bottom left','bottom in the middle','bottom right']
        location = st.selectbox('select the photo location on page',location_options)

        location_vector = [1 if location == loc else 0 for loc in location_options]  


        input_dt = pd.DataFrame([  
            page2_cm_encoded +  # 8 values
            colour_encoded +  # 1 value
            price_scaled +  # 1 value
            page_number_encode +  # 1 value
            page1_main_category_vector +  # 4 values
            location_vector +  # 6 values
            price_2_vector +  # 2 values
            model_photo_vector  # 2 values
        ], columns=[
            'page2_clothing_model_0', 'page2_clothing_model_1',
            'page2_clothing_model_2', 'page2_clothing_model_3',
            'page2_clothing_model_4', 'page2_clothing_model_5',
            'page2_clothing_model_6', 'page2_clothing_model_7',
            'colour', 'price', 'page',
            'page1_main_category_1', 'page1_main_category_2',
            'page1_main_category_3', 'page1_main_category_4',
            'location_1', 'location_2', 'location_3',
            'location_4', 'location_5', 'location_6',
            'price_2_1', 'price_2_2',
            'model_photography_1', 'model_photography_2'
        ])
        
        if st.button("Predict"):
            cluster_pred = kmeans.predict(input_dt)[0]
        
            st.success(f"As per the given browsing pattern the Customer belongs to {cluster_pred} cluster")

    