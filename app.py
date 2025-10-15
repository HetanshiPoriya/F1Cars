import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from PIL import Image # Pillow is used here to ensure images are handled correctly

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="F1 Race Predictor 🏎️")

# --- Load Model and Preprocessor (Cached for Efficiency) ---
@st.cache_resource
def load_resources():
    """Loads the pre-trained model and preprocessor from disk."""
    try:
        model = joblib.load('F1_Racing.joblib')
        preprocessor = joblib.load('F1_Racing_preprocessor.joblib')
        return model, preprocessor
    except FileNotFoundError:
        st.error("Error: Required model files ('F1_Racing.joblib', 'F1_Racing_preprocessor.joblib') not found. Please ensure they are in the same directory.")
        return None, None

# --- Get User Input ---
def get_user_input():
    """Collects user inputs from Streamlit widgets and returns a DataFrame."""
    
    driver_list = ['Lewis Hamilton', 'Max Verstappen', 'Charles Leclerc', 'Lando Norris', 'Carlos Sainz', 'Sergio Perez']
    constructor_list = ['Mercedes', 'Red Bull Racing', 'Ferrari', 'McLaren', 'Aston Martin']
    circuit_list = ['Bahrain Grand Prix', 'Monaco Grand Prix', 'Silverstone', 'Spa-Francorchamps', 'Monza']
    
    with st.sidebar:
        st.header("🏁 Race & Driver Inputs")
        
        year = st.number_input("Year", min_value=1950, max_value=2024, value=2024, step=1)
        grid = st.number_input("Starting Grid Position", min_value=1, max_value=22, value=1, step=1)
        race_round = st.number_input("Race Round", min_value=1, max_value=24, value=1, step=1)
        laps = st.number_input("Laps", min_value=10, max_value=100, value=60, step=1)
        
        driver = st.selectbox("Driver", driver_list)
        constructor_name = st.selectbox("Constructor", constructor_list)
        circuit_name = st.selectbox("Circuit", circuit_list)
        
    user_data = {
        'year': [year],
        'round': [race_round],
        'grid': [grid],
        'laps': [laps],
        'driver': [driver],
        'constructor_name': [constructor_name],
        'circuit_name': [circuit_name]
    }
    
    return pd.DataFrame(user_data)

# --- Feature Importance Plotting Function ---
def plot_feature_importance(model, preprocessor):
    """
    Creates a Plotly bar chart of feature importances by aligning
    model importances with preprocessor feature names.
    """
    try:
        all_feature_names = preprocessor.get_feature_names_out()
        feature_importances = model.feature_importances_

        if len(all_feature_names) != len(feature_importances):
            raise ValueError(f"Feature name count ({len(all_feature_names)}) does not match importance count ({len(feature_importances)}).")

        feature_importance_df = pd.DataFrame({
            'Feature': all_feature_names,
            'Importance': feature_importances
        })

        fi_sorted = feature_importance_df.sort_values(by='Importance', ascending=False).head(20).sort_values(by='Importance', ascending=True)

        fig = px.bar(
            fi_sorted,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Top 20 Feature Importance (Mean Decrease Impurity)",
            labels={
                "Importance": "Feature Importance Score",
                "Feature": "Variable"
            },
            template="plotly_white",
            height=700,
            color_discrete_sequence=['#E10600']
        )

        fig.update_layout(
            xaxis_title="Feature Importance Score",
            yaxis_title="Variable",
            xaxis_tickformat=".3f"
        )
        return fig
        
    except Exception as e:
        st.warning(f"Could not generate Feature Importance Plot. Error: {e}")
        return None

# --- Main App Execution ---
def main():
    # 1. Load Resources
    model, preprocessor = load_resources()

    # 2. Page Layout and Title
    st.markdown(
        """
        <div style="text-align: center; padding-bottom: 20px;">
            <h1 style='color: #E10600; font-weight: 900;'>F1 Race Prediction Dashboard</h1>
            <h3 style='color: #666;'>Predict Race Outcome & Analyze Model Drivers</h3>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # 🏎️ ACCESSING THE LOCAL .JPG FILE
    # Make sure your image file is named 'f1_car.jpg' and is in the same directory as app.py
    image_path = 'f1_car.jpg'
    
    # Use st.image() to display the local image
    try:
        # We load the image using its path
        f1_image = Image.open(image_path)
        st.image(f1_image, use_container_width=True, caption="Visualizing the Race to the Finish")
    except FileNotFoundError:
        # This warning will display if the file is missing or misspelled
        st.warning(f"Image file not found: {image_path}. Please make sure the file exists in the same directory.")
        
    if model is not None and preprocessor is not None:
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Predict Race Outcome")
            user_input_df = get_user_input()
            
            if st.button("Predict Final Position", use_container_width=True, type="primary"):
                with st.spinner("Calculating prediction..."):
                    try:
                        transformed_input = preprocessor.transform(user_input_df)
                        prediction = model.predict(transformed_input)
                        
                        st.subheader("Predicted Finish Position")
                        st.balloons()
                        st.markdown(f"<div style='background-color: #E10600; color: white; padding: 10px; border-radius: 5px; text-align: center; font-size: 24px;'>🏆 Position: <strong>{int(prediction[0])}</strong></div>", unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"An error occurred during prediction. Check console for details: {e}")
                        
        with col2:
            st.subheader("Model Insights")
            
            fig = plot_feature_importance(model, preprocessor)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("Cannot run app. Please ensure model and preprocessor files are available.")
        
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #888;'>Data Science in the Fast Lane 🏎️</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()