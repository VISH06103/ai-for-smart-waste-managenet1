import os
import io
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from groq import Groq
import streamlit as st

# ==========================================
# 1. COMPUTER VISION & CLASSIFIER MODULE
# ==========================================
CLASSES = ["Organic", "Plastic", "Recyclable", "Hazardous"]

class WasteClassifier:
    def __init__(self):
        # MobileNetV2 lightweight base model initialized for inference
        base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
        output = tf.keras.layers.Dense(len(CLASSES), activation="softmax")(x)
        self.model = tf.keras.Model(inputs=base.input, outputs=output)

    def classify(self, image_bytes: bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # OpenCV Preprocessing: Color conversion, Gaussian noise reduction, resizing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        blurred = cv2.GaussianBlur(img_rgb, (5, 5), 0)
        resized = cv2.resize(blurred, (224, 224))
        
        tensor = preprocess_input(np.expand_dims(resized.astype(np.float32), axis=0))
        preds = self.model.predict(tensor, verbose=0)[0]
        
        idx = int(np.argmax(preds))
        return {
            "category": CLASSES[idx],
            "confidence": float(preds[idx]),
            "probabilities": {CLASSES[i]: float(preds[i]) for i in range(len(CLASSES))}
        }

# ==========================================
# 2. PRIORITY ROUTE OPTIMIZATION MODULE
# ==========================================
URGENCY_MULTIPLIERS = {
    "Hazardous": 1.5,
    "Organic": 1.3,
    "Plastic": 1.0,
    "Recyclable": 1.0
}

def haversine_distance(coord1: tuple, coord2: tuple) -> float:
    """Calculates distance in kilometers between two lat/lon pairs."""
    R = 6371.0
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return R * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))

def optimize_collection_route(depot_location: tuple, bins: list) -> dict:
    """Priority-weighted Nearest Neighbor heuristic route planner."""
    eligible_bins = [b for b in bins if b["fill_level"] >= 60.0 or b["waste_type"] == "Hazardous"]
    
    if not eligible_bins:
        return {"route_order": [], "detailed_route": [], "total_distance_km": 0.0}

    for b in eligible_bins:
        multiplier = URGENCY_MULTIPLIERS.get(b["waste_type"], 1.0)
        b["priority"] = b["fill_level"] * multiplier

    unvisited = eligible_bins.copy()
    current_pos = depot_location
    route = []
    total_distance = 0.0

    while unvisited:
        # Pick next bin maximizing priority over distance penalty
        next_bin = max(
            unvisited, 
            key=lambda item: item["priority"] / (haversine_distance(current_pos, (item["lat"], item["lon"])) + 0.1)
        )
        dist = haversine_distance(current_pos, (next_bin["lat"], next_bin["lon"]))
        total_distance += dist
        current_pos = (next_bin["lat"], next_bin["lon"])
        route.append(next_bin)
        unvisited.remove(next_bin)

    total_distance += haversine_distance(current_pos, depot_location)

    return {
        "route_order": [b["bin_id"] for b in route],
        "detailed_route": route,
        "total_distance_km": round(total_distance, 2)
    }

# ==========================================
# 3. AGENTIC AI DISPATCHER (GROQ API)
# ==========================================
class WasteManagementAgent:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if self.api_key:
            self.client = Groq(api_key=self.api_key)
        else:
            self.client = None

    def analyze_fleet_status(self, bin_data: list, route_summary: dict) -> str:
        if not self.client:
            return "⚠️ **Groq API Key Missing.** Set `GROQ_API_KEY` in Streamlit Secrets or sidebar."
        
        prompt = f"""
        You are an AI Operational Dispatcher for a Smart City Waste System.
        Analyze this telemetry and collection route, then generate a concise executive advisory report:
        
        Bin Telemetry: {bin_data}
        Calculated Route: {route_summary}
        
        Formatting Requirements:
        1. 🚨 Critical Safety/Hazard Warnings
        2. 🚛 Fleet & Route Optimization Recommendations
        3. 📋 Action Items for Drivers
        """
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a professional AI environmental logistics officer."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=600
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"❌ Error contacting Groq API: {str(e)}"

# ==========================================
# 4. STREAMLIT USER INTERFACE
# ==========================================
st.set_page_config(page_title="Smart Waste Ops Center", layout="wide")

# Retrieve API Key from Secrets or Sidebar Input
groq_key = st.secrets.get("GROQ_API_KEY", "")
if not groq_key:
    groq_key = st.sidebar.text_input("Enter Groq API Key:", type="password")

@st.cache_resource
def load_classifier():
    return WasteClassifier()

classifier = load_classifier()

st.title("♻️ Smart Waste Management System")

tab1, tab2, tab3 = st.tabs(["📸 Camera Inspection", "🗺️ Route Optimization", "🤖 Groq Agentic Ops"])

# --- TAB 1: CLASSIFICATION ---
with tab1:
    st.header("Waste Image Classification")
    uploaded_file = st.file_uploader("Upload waste image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        col1, col2 = st.columns(2)
        image = Image.open(uploaded_file)
        col1.image(image, caption="Uploaded Sample", use_container_width=True)
        
        if col1.button("Classify Waste"):
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            res = classifier.classify(img_byte_arr.getvalue())
            
            col2.success(f"**Category:** {res['category']}")
            col2.info(f"**Confidence:** {res['confidence'] * 100:.2f}%")
            col2.write("Probability Breakdown:")
            col2.json(res["probabilities"])

# --- TAB 2: ROUTE OPTIMIZATION ---
with tab2:
    st.header("Priority Route Optimizer")
    
    mock_bins = [
        {"bin_id": "BIN-101", "lat": 12.9716, "lon": 77.5946, "fill_level": 85.0, "waste_type": "Organic"},
        {"bin_id": "BIN-102", "lat": 12.9750, "lon": 77.6000, "fill_level": 92.0, "waste_type": "Hazardous"},
        {"bin_id": "BIN-103", "lat": 12.9650, "lon": 77.5800, "fill_level": 40.0, "waste_type": "Plastic"},
        {"bin_id": "BIN-104", "lat": 12.9800, "lon": 77.6100, "fill_level": 78.0, "waste_type": "Recyclable"}
    ]
    
    st.dataframe(pd.DataFrame(mock_bins), use_container_width=True)
    
    if st.button("Calculate Priority Route"):
        route_data = optimize_collection_route((12.9700, 77.5900), mock_bins)
        st.success(f"Optimized Route Distance: **{route_data['total_distance_km']} km**")
        st.write(" **Sequence:** " + " ➔ ".join(["Depot"] + route_data["route_order"] + ["Depot"]))
        
        map_df = pd.DataFrame(route_data["detailed_route"])
        if not map_df.empty:
            st.map(map_df[["lat", "lon"]])

# --- TAB 3: GROQ AGENTIC AI ---
with tab3:
    st.header("Groq Llama-3 Operational Advisory")
    if st.button("Run AI Agent Assessment"):
        with st.spinner("Analyzing fleet telemetry with Groq..."):
            agent = WasteManagementAgent(api_key=groq_key)
            route_data = optimize_collection_route((12.9700, 77.5900), mock_bins)
            report = agent.analyze_fleet_status(mock_bins, route_data)
            st.markdown(report)
