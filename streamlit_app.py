import json
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


def main():
    st.title("End to End Customer Satisfaction Pipeline with ZenML")

    high_level_image = Image.open("_assets/high_level_overview.png")
    st.image(high_level_image, caption="High Level Pipeline")

    whole_pipeline_image = Image.open("_assets/training_and_deployment_pipeline_updated.png")

    st.markdown(
        """ 
    #### Problem Statement 
     The objective here is to predict the customer satisfaction score for a given order based on features like order status, price, payment, etc. I will be using [ZenML](https://zenml.io/) to build a production-ready pipeline to predict the customer satisfaction score for the next order or purchase.    """
    )
    st.image(whole_pipeline_image, caption="Whole Pipeline")
    st.markdown(
        """ 
    Above is a figure of the whole pipeline, we first ingest the data, clean it, train the model, and evaluate the model, and if data source changes or any hyperparameter values changes, deployment will be triggered, and (re) trains the model and if the model meets minimum accuracy requirement, the model will be deployed.
    """
    )

    st.markdown(
        """ 
    #### Description of Features 
    This app is designed to predict the customer satisfaction score for a given customer. You can input the features of the product listed below and get the customer satisfaction score. 
    | Models        | Description   | 
    | ------------- | -     | 
    | Payment Sequential | Customer may pay an order with more than one payment method. If he does so, a sequence will be created to accommodate all payments. | 
    | Payment Installments   | Number of installments chosen by the customer. |  
    | Payment Value |       Total amount paid by the customer. | 
    | Price |       Price of the product. |
    | Freight Value |    Freight value of the product.  | 
    | Product Name length |    Length of the product name. |
    | Product Description length |    Length of the product description. |
    | Product photos Quantity |    Number of product published photos |
    | Product weight measured in grams |    Weight of the product measured in grams. | 
    | Product length (CMs) |    Length of the product measured in centimeters. |
    | Product height (CMs) |    Height of the product measured in centimeters. |
    | Product width (CMs) |    Width of the product measured in centimeters. |
    """
    )
    payment_sequential = st.sidebar.slider("Payment Sequential", min_value=0, max_value=10, value=0)
    payment_installments = st.sidebar.slider("Payment Installments", min_value=0, max_value=10, value=0)
    payment_value = st.number_input("Payment Value", min_value=0.0, value=0.0)
    price = st.number_input("Price", min_value=0.0, value=0.0)
    freight_value = st.number_input("Freight Value", min_value=0.0, value=0.0)
    product_name_length = st.number_input("Product Name Length", min_value=0, value=0)
    product_description_length = st.number_input("Product Description Length", min_value=0, value=0)
    product_photos_qty = st.number_input("Product Photos Quantity", min_value=0, value=0)
    product_weight_g = st.number_input("Product Weight (grams)", min_value=0, value=0)
    product_length_cm = st.number_input("Product Length (cm)", min_value=0, value=0)
    product_height_cm = st.number_input("Product Height (cm)", min_value=0, value=0)
    product_width_cm = st.number_input("Product Width (cm)", min_value=0, value=0)

    if st.button("Predict"):
        model_path = "D:/GitHub/customer-satisfaction/saved_model/model.pkl"
        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        df = pd.DataFrame(
            {
                "payment_sequential": [payment_sequential],
                "payment_installments": [payment_installments],
                "payment_value": [payment_value],
                "price": [price],
                "freight_value": [freight_value],
                "product_name_lenght": [product_name_length],
                "product_description_lenght": [product_description_length],
                "product_photos_qty": [product_photos_qty],
                "product_weight_g": [product_weight_g],
                "product_length_cm": [product_length_cm],
                "product_height_cm": [product_height_cm],
                "product_width_cm": [product_width_cm],
            }
        )
        data = df.values
        try:
            pred = model.predict(data)
            st.success(
                "Your Customer Satisfactory rate(range between 0 - 5) with given product details is :-{}".format(
                    pred[0]
                )
            )
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    if st.button("Results"):
        st.write(
            "We have experimented with two ensemble and tree based models and compared the performance of each model. The results are as follows:"
        )

        df = pd.DataFrame(
            {
                "Models": ["LightGBM", "Xgboost"],
                "MSE": [1.804, 1.781],
                "RMSE": [1.343, 1.335],
            }
        )
        st.dataframe(df)

        st.write(
            "Following figure shows how important each feature is in the model that contributes to the target variable or contributes in predicting customer satisfaction rate."
        )
        image = Image.open("_assets/feature_importance_gain.png")
        st.image(image, caption="Feature Importance Gain")


if __name__ == "__main__":
    main()
