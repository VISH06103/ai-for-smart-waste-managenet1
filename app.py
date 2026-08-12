import streamlit as st
from groq import Groq
from PIL import Image
import base64
import json
import io
import pandas as pd
import plotly.express as px
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="AI Smart Waste Management System",
    page_icon="♻️",
    layout="wide"
)

# Initialize Session State
if "waste_logs" not in st.session_state:
    st.session_state.waste_logs = []

# Helper: Encode PIL image to Base64
def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Helper: Analyze image via Groq API
def analyze_waste(image, api_key):
    client = Groq(api_key=api_key)
    base64_image = encode_image(image)
    
    prompt = """
    Analyze this waste image. Return ONLY a valid JSON object with the following structure:
    {
      "category": "Plastic" or "Organic" or "Recyclable" or "Hazardous",
      "item_identified": "Short name of object",
      "confidence": percentage number between 80 and 99,
      "estimated_volume_liters": estimated numeric volume in liters (e.g. 1.5),
      "hazard_level": "Low" or "Medium" or "High",
      "recyclable": true or false,
      "disposal_instructions": "Brief step-by-step handling instruction"
    }
    """
    
    response = client.chat.completions.create(
        model="qwen/qwen3.6-27b",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        temperature=0.2,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

# Route Optimization Logic (Greedy Priority Algorithm)
def optimize_collection_routes(logs):
    if not logs:
        return None
    
    df = pd.DataFrame(logs)
    np.random.seed(42)
    
    # Base location (Depot)
    base_lat, base_lon = 37.7749, -122.4194
    
    df["lat"] = base_lat + np.random.uniform(-0.05, 0.05, len(df))
    df["lon"] = base_lon + np.random.uniform(-0.05, 0.05, len(df))
    
    # Priority Score = Volume * Hazard Weight
    hazard_weight = {"Low": 1, "Medium": 2, "High": 3}
    df["priority_score"] = df.apply(
        lambda r: r["estimated_volume_liters"] * hazard_weight.get(r["hazard_level"], 1), axis=1
    )
    
    # Sort route by Priority Score
    optimized_df = df.sort_values(by="priority_score", ascending=False).reset_index(drop=True)
    optimized_df["Pickup Stop"] = optimized_df.index + 1
    return optimized_df

# Sidebar UI
st.sidebar.title("⚙️ Configuration")
groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

st.sidebar.markdown("---")
st.sidebar.info("Upload waste images to automatically classify waste, calculate volume, optimize pickup routes, and view analytics.")

# Main Application Title
st.title("♻️ AI Smart Waste Management System")
st.write("Automated AI Waste Categorization, Volume Estimation & Route Optimization")

# Tab Layout
tab1, tab2, tab3 = st.tabs(["📸 Waste Classifier", "🗺️ Route Optimizer", "📊 Analytics Dashboard"])

# TAB 1: CLASSIFIER & ANALYSIS
with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Waste Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            if not groq_api_key:
                st.warning("⚠️ Please enter your Groq API key in the sidebar to analyze.")
            else:
                if st.button("🔍 Analyze Waste", type="primary"):
                    with st.spinner("AI Agent analyzing image..."):
                        try:
                            result = analyze_waste(image, groq_api_key)
                            
                            log_entry = {
                                "id": f"BIN-{len(st.session_state.waste_logs) + 101}",
                                "item": result.get("item_identified", "Unknown"),
                                "category": result.get("category", "General"),
                                "confidence": result.get("confidence", 90),
                                "estimated_volume_liters": float(result.get("estimated_volume_liters", 1.0)),
                                "hazard_level": result.get("hazard_level", "Low"),
                                "recyclable": result.get("recyclable", False),
                                "disposal_instructions": result.get("disposal_instructions", "Standard disposal.")
                            }
                            st.session_state.waste_logs.append(log_entry)
                            st.success("Analysis Complete!")
                        except Exception as e:
                            st.error(f"Error analyzing image: {str(e)}")

    with col2:
        st.subheader("Analysis Results")
        if st.session_state.waste_logs:
            latest = st.session_state.waste_logs[-1]
            
            st.metric("Detected Item", latest["item"])
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Category", latest["category"])
            c2.metric("Volume (L)", f"{latest['estimated_volume_liters']} L")
            c3.metric("Hazard Level", latest["hazard_level"])
            
            st.markdown(f"**Recyclable:** {'Yes 🟢' if latest['recyclable'] else 'No 🔴'}")
            st.markdown(f"**AI Confidence:** {latest['confidence']}%")
            
            st.info(f"**Disposal Instructions:**\n\n{latest['disposal_instructions']}")
        else:
            st.info("Upload an image and click 'Analyze Waste' to see results.")

# TAB 2: ROUTE OPTIMIZATION
with tab2:
    st.subheader("🚛 Waste Collection Route Optimization")
    
    if not st.session_state.waste_logs:
        st.warning("No waste data available yet. Classify some items in the 'Waste Classifier' tab first.")
    else:
        optimized_data = optimize_collection_routes(st.session_state.waste_logs)
        
        col_map, col_table = st.columns([1.2, 1])
        
        with col_map:
            st.markdown("#### Pickup Map (Priority-Based Path)")
            fig = px.scatter_mapbox(
                optimized_data,
                lat="lat",
                lon="lon",
                hover_name="item",
                hover_data=["category", "estimated_volume_liters", "hazard_level", "Pickup Stop"],
                color="category",
                size="estimated_volume_liters",
                zoom=11,
                height=450
            )
            fig.update_layout(mapbox_style="open-street-map")
            st.plotly_chart(fig, use_container_width=True)
            
        with col_table:
            st.markdown("#### Optimized Pickup Sequence")
            st.dataframe(
                optimized_data[["Pickup Stop", "id", "item", "category", "estimated_volume_liters", "hazard_level"]],
                use_container_width=True,
                hide_index=True
            )

# TAB 3: ANALYTICS DASHBOARD
with tab3:
    st.subheader("📈 Waste Management Analytics")
    
    if not st.session_state.waste_logs:
        st.warning("No data recorded yet. Upload images to populate the dashboard.")
    else:
        df_logs = pd.DataFrame(st.session_state.waste_logs)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Items Processed", len(df_logs))
        m2.metric("Total Volume Collected", f"{df_logs['estimated_volume_liters'].sum():.1f} L")
        m3.metric("Recyclable Ratio", f"{(df_logs['recyclable'].mean() * 100):.1f}%")
        m4.metric("High Hazard Bins", len(df_logs[df_logs["hazard_level"] == "High"]))
        
        st.markdown("---")
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("#### Waste Distribution by Category")
            fig_pie = px.pie(df_logs, names="category", title="Waste Types Breakdown", hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with c2:
            st.markdown("#### Total Volume by Category (Liters)")
            fig_bar = px.bar(
                df_logs.groupby("category")["estimated_volume_liters"].sum().reset_index(),
                x="category",
                y="estimated_volume_liters",
                color="category",
                labels={"estimated_volume_liters": "Volume (L)", "category": "Category"}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
