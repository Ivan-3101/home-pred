import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import os
import zipfile
from dotenv import load_dotenv
import plotly.graph_objects as go
import requests
from PIL import Image
from io import BytesIO

# Load environment variables
load_dotenv()

# Function to unzip and load the model
@st.cache_resource
def load_model():
    zip_file = 'food101_mobilenetv2.zip'
    model_dir = 'food101_mobilenetv2'
    
    # Unzip the file if it's not already extracted
    if not os.path.exists(model_dir):
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(model_dir)
    
    model_path = os.path.join(model_dir, 'food101_mobilenetv2.h5')  # Adjust the path if necessary
    return tf.keras.models.load_model(model_path)

model = load_model()

# List of Food-101 classes
food_classes = [
    'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
    'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
    'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
    'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla',
    'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder',
    'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes',
    'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots',
    'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries',
    'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice', 'frozen_yogurt',
    'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon',
    'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros',
    'hummus', 'ice_cream', 'lasagna', 'lobster_bisque', 'lobster_roll_sandwich',
    'macaroni_and_cheese', 'macarons', 'miso_soup', 'mussels', 'nachos', 'omelette',
    'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck',
    'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich', 'ramen',
    'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad',
    'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls',
    'steak', 'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare',
    'waffles'
]
IMG_SIZE = 224  # Image size expected by the model

# Custom CSS to enhance the app's appearance with dark theme
st.markdown("""
<style>
    .reportview-container {
        background: #0E1117;
        color: #FAFAFA;
    }
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    h1, h2, h3 {
        color: #FAFAFA;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        font-weight: bold;
        padding: 10px 20px;
    }
    .stTextInput>div>div>input {
        background-color: #262730;
        color: #FAFAFA;
        border-color: #4CAF50;
        border-radius: 5px;
    }
    .stPlotlyChart {
        background-color: #262730;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #262730;
        border-radius: 4px 4px 0 0;
        padding: 10px 24px;
        color: #FAFAFA;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

def preprocess_image(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype='float32')
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image
    return img_array

def predict_food(img):
    preprocessed_img = preprocess_image(img)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_food_name = food_classes[predicted_class_index]
    confidence = predictions[0][predicted_class_index]
    return predicted_food_name, confidence

def call_food_analysis_api(food_name):
    api_url = "https://home-api-z0vu.onrender.com/analyze_food"  # Flask API URL
    headers = {'Content-Type': 'application/json'}
    data = {"food_name": food_name}

    response = requests.post(api_url, json=data, headers=headers)

    if response.status_code == 200:
        return response.json()['analysis']
    else:
        st.error(f"Error: {response.json().get('error', 'Unknown error occurred.')}")
        return None

def parse_analysis(analysis):
    lines = analysis.split('\n')
    ingredients = lines[0].split(': ')[1].split('|')
    health_scores = [int(score) for score in lines[1].split(': ')[1].split('|')]
    overall_health = lines[2].split(': ')[1]
    return ingredients, health_scores, overall_health

def create_health_chart(ingredients, health_scores):
    colors = ['#EF4444' if score < 4 else '#F59E0B' if score < 7 else '#10B981' for score in health_scores]
    fig = go.Figure(data=[go.Bar(
        x=ingredients,
        y=health_scores,
        marker_color=colors,
        text=health_scores,
        textposition='auto',
    )])
    fig.update_layout(
        title={
            'text': "Ingredient Health Scores",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Ingredients",
        yaxis_title="Health Score (0-10)",
        yaxis=dict(range=[0, 10]),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial", size=14, color="#FAFAFA"),
    )
    return fig

def get_food_image(food_name):
    api_key = "eA4LaKMKCBIKFvv6ktiOBaxBTFmA9BMzybaJUoHAWSotmh0agkhF0wzQ"
    if not api_key:
        st.warning("Pexels API key not found. Please check your .env file.")
        return None

    url = f"https://api.pexels.com/v1/search?query={food_name}&per_page=1"
    headers = {"Authorization": api_key}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        data = response.json()
        if data['photos']:
            image_url = data['photos'][0]['src']['medium']
            image_response = requests.get(image_url)
            image_response.raise_for_status()
            img = Image.open(BytesIO(image_response.content))
            return img
        else:
            st.info(f"No images found for '{food_name}' on Pexels.")
            return None
    except requests.RequestException as e:
        st.error(f"Error retrieving image: {str(e)}")
        return None

def get_health_category(score):
    if score >= 7:
        return "Healthy", "#10B981"
    elif 4 <= score < 7:
        return "Moderate", "#F59E0B"
    else:
        return "Unhealthy", "#EF4444"

def display_food_analysis(food_name):
    st.subheader(f"Analyzing: {food_name}")
    
    with st.spinner("Analyzing food safety and health risks..."):
        analysis = call_food_analysis_api(food_name)
    
    if analysis:
        ingredients, health_scores, overall_health = parse_analysis(analysis)

        fig = create_health_chart(ingredients, health_scores)
        st.plotly_chart(fig, use_container_width=True)

        # Handle case where overall_health is not a number
        try:
            overall_health = int(overall_health)
            overall_category, color = get_health_category(overall_health)
            st.markdown(f"<h3 style='color:{color};text-align:center;'>Overall Health: {overall_category} ({overall_health}/10)</h3>", unsafe_allow_html=True)
        except ValueError:
            st.markdown(f"<h3 style='color:#EF4444;text-align:center;'>Overall Health: {overall_health}</h3>", unsafe_allow_html=True)


# Streamlit app structure
st.title("Food Recognition and Health Analyzer")

uploaded_file = st.file_uploader("Upload an image of food", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Predict the food
    predicted_food_name, confidence = predict_food(img)
    st.write(f"**Predicted food:** {predicted_food_name} (Confidence: {confidence * 100:.2f}%)")

    # Display food image from Pexels
    food_image = get_food_image(predicted_food_name)
    if food_image:
        st.image(food_image, caption=f'{predicted_food_name} from Pexels', use_column_width=True)

    # Display food analysis
    display_food_analysis(predicted_food_name)
