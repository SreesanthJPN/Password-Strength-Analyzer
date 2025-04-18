import streamlit as st
import torch
import numpy as np
from feature_extractor import extract_password_features
from model import PasswordNet
import math
from datetime import datetime, timedelta

# Set page config with minimal theme
st.set_page_config(
    page_title="Password Strength Classifier",
    page_icon="🔒",
    layout="centered"
)

# Enhanced Minimalist CSS
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

    /* Enhanced color scheme */
    :root {
        --primary: #111827;
        --secondary: #4B5563;
        --background: #F9FAFB;
        --border: #E5E7EB;
        --card: #FFFFFF;
        --success: #059669;
        --warning: #D97706;
        --danger: #DC2626;
    }

    /* Base styles */
    body {
        font-family: 'Inter', sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    .main {
        background: var(--background);
        max-width: 800px;
        margin: 0 auto;
        padding: 3rem 2rem;
    }

    /* Enhanced input styling */
    .stTextInput>div>div>input {
        background-color: var(--card);
        border-radius: 8px;
        padding: 14px 16px;
        border: 1px solid var(--border);
        font-size: 16px;
        font-weight: 500;
        color: var(--primary);
        transition: all 0.2s ease;
    }

    .stTextInput>div>div>input:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 2px rgba(17, 24, 39, 0.1);
        outline: none;
    }

    .stTextInput>div>div>input::placeholder {
        color: var(--secondary);
        opacity: 0.6;
    }

    /* Enhanced metric cards */
    .stMetric {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 20px;
        transition: transform 0.2s ease;
    }

    .stMetric:hover {
        transform: translateY(-2px);
    }

    .stMetric [data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: 600;
        color: var(--primary);
    }

    .stMetric [data-testid="stMetricLabel"] {
        font-size: 14px;
        font-weight: 500;
        color: var(--secondary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Enhanced feature boxes */
    .feature-box {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 16px 20px;
        margin: 12px 0;
        transition: all 0.2s ease;
    }

    .feature-box:hover {
        border-color: var(--primary);
    }

    /* Enhanced strength indicator */
    .strength-indicator {
        padding: 24px;
        border-radius: 8px;
        margin: 24px 0;
        text-align: center;
        font-size: 28px;
        font-weight: 600;
        letter-spacing: 0.5px;
        border: 1px solid var(--border);
        background: var(--card);
        transition: all 0.2s ease;
    }

    .strength-indicator.weak { color: var(--danger); border-color: var(--danger); }
    .strength-indicator.good { color: var(--warning); border-color: var(--warning); }
    .strength-indicator.strong { color: var(--success); border-color: var(--success); }

    /* Enhanced tips boxes */
    .tips-box {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 24px;
        margin: 24px 0;
    }

    .tips-box h3 {
        font-size: 18px;
        font-weight: 600;
        color: var(--primary);
        margin-bottom: 16px;
        letter-spacing: 0.5px;
    }

    .tips-box ul {
        list-style-type: none;
        padding: 0;
        margin: 0;
    }

    .tips-box li {
        padding: 8px 0;
        color: var(--secondary);
        font-size: 15px;
        line-height: 1.6;
        position: relative;
        padding-left: 24px;
    }

    .tips-box li:before {
        content: "•";
        color: var(--primary);
        position: absolute;
        left: 0;
        font-size: 20px;
        line-height: 1;
    }

    /* Enhanced header */
    .header {
        text-align: center;
        margin-bottom: 3rem;
    }

    .header h1 {
        font-size: 2.5rem;
        font-weight: 600;
        margin: 0;
        color: var(--primary);
        letter-spacing: -0.5px;
    }

    .header p {
        font-size: 1.1rem;
        color: var(--secondary);
        margin: 12px 0 0 0;
        font-weight: 400;
    }

    /* Enhanced footer */
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 2rem;
        border-top: 1px solid var(--border);
        color: var(--secondary);
        font-size: 14px;
        letter-spacing: 0.5px;
    }

    /* Section headers */
    .stSubheader {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--primary);
        margin: 2rem 0 1rem 0;
        letter-spacing: -0.5px;
    }

    /* Remove Streamlit default styling */
    .stApp {
        background: var(--background);
    }

    .stButton>button {
        border-radius: 8px;
        font-weight: 500;
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--primary);
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    model = PasswordNet(input_size=13)
    model.load_state_dict(torch.load('Password_strength_model.pth'))
    model.eval()
    return model

# Function to predict password strength
def predict_strength(password):
    features = extract_password_features(password)
    feature_values = list(features.values())
    input_tensor = torch.FloatTensor(feature_values).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()
    strength_levels = {0: "Weak", 1: "Good", 2: "Strong"}
    return strength_levels[prediction], output.numpy()[0]

def calculate_crack_time(password):
    """
    Calculate estimated time to crack password based on entropy and common cracking speeds
    """
    # Character sets
    lowercase = 26
    uppercase = 26
    numbers = 10
    special = 33  # Common special characters
    
    # Count character types
    has_lower = any(c.islower() for c in password)
    has_upper = any(c.isupper() for c in password)
    has_num = any(c.isdigit() for c in password)
    has_special = any(not c.isalnum() for c in password)
    
    # Calculate possible combinations
    charset_size = 0
    if has_lower: charset_size += lowercase
    if has_upper: charset_size += uppercase
    if has_num: charset_size += numbers
    if has_special: charset_size += special
    
    # Calculate entropy
    entropy = len(password) * math.log2(charset_size)
    
    # Common cracking speeds (guesses per second)
    # These are rough estimates based on different attack methods
    speeds = {
        'online': 1,  # Online attack
        'offline': 1000000000,  # Offline attack with GPU
        'brute_force': 1000000000000  # Brute force with supercomputer
    }
    
    # Calculate time for each method
    times = {}
    for method, speed in speeds.items():
        guesses = 2 ** entropy
        seconds = guesses / speed
        
        # Convert to human readable time
        if seconds < 60:
            times[method] = f"{seconds:.2f} seconds"
        elif seconds < 3600:
            times[method] = f"{seconds/60:.2f} minutes"
        elif seconds < 86400:
            times[method] = f"{seconds/3600:.2f} hours"
        elif seconds < 31536000:  # 1 year
            times[method] = f"{seconds/86400:.2f} days"
        else:
            years = seconds / 31536000
            if years < 1000:
                times[method] = f"{years:.2f} years"
            else:
                times[method] = f"{years/1000:.2f} millennia"
    
    return times, entropy

# Load the model
model = load_model()

# Enhanced header
st.markdown("""
    <div class='header'>
        <h1>Password Strength</h1>
        <p>Analyze your password's security</p>
    </div>
""", unsafe_allow_html=True)

# Main container
with st.container():
    # Password input with larger styling
    st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h2 style='font-size: 1.5rem; font-weight: 500; color: var(--secondary); margin-bottom: 1.5rem;'>
                Enter your password to analyze
            </h2>
        </div>
    """, unsafe_allow_html=True)
    password = st.text_input("", type="password", placeholder="Enter password")
    
    if password:
        # Get prediction
        strength, probabilities = predict_strength(password)
        
        # Calculate crack time
        crack_times, entropy = calculate_crack_time(password)
        
        # Strength indicator with enhanced styling
        strength_class = strength.lower()
        st.markdown(f"""
            <div class='strength-indicator {strength_class}'>
                {strength}
            </div>
        """, unsafe_allow_html=True)
        
        # Display crack time information
        st.markdown("""
            <div class='feature-box' style='margin-top: 1rem;'>
                <h3 style='margin: 0 0 1rem 0; color: var(--primary);'>Password Security Analysis</h3>
                <p style='margin: 0.5rem 0;'><strong>Entropy:</strong> {entropy:.2f} bits</p>
                <p style='margin: 0.5rem 0;'><strong>Online Attack:</strong> {online_time}</p>
                <p style='margin: 0.5rem 0;'><strong>Offline Attack:</strong> {offline_time}</p>
                <p style='margin: 0.5rem 0;'><strong>Brute Force:</strong> {brute_time}</p>
            </div>
        """.format(
            entropy=entropy,
            online_time=crack_times['online'],
            offline_time=crack_times['offline'],
            brute_time=crack_times['brute_force']
        ), unsafe_allow_html=True)
        
        # Confidence distribution
        st.subheader("Confidence")
        cols = st.columns(3)
        with cols[0]:
            st.metric("Weak", f"{probabilities[0]*100:.1f}%")
        with cols[1]:
            st.metric("Good", f"{probabilities[1]*100:.1f}%")
        with cols[2]:
            st.metric("Strong", f"{probabilities[2]*100:.1f}%")
        
        # Feature analysis
        st.subheader("Analysis")
        features = extract_password_features(password)
        for feature, value in features.items():
            st.markdown(f"""
                <div class='feature-box'>
                    {feature.replace('_', ' ').title()}: {value}
                </div>
            """, unsafe_allow_html=True)

    # Tips section
    st.markdown("""
        <div class='tips-box'>
            <h3>Tips</h3>
            <ul>
                <li>Use at least 12 characters</li>
                <li>Mix uppercase and lowercase letters</li>
                <li>Include numbers and special characters</li>
                <li>Avoid common words and patterns</li>
                <li>Don't use personal information</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# Enhanced footer
st.markdown("""
    <div class='footer'>
        Password Strength Classifier
    </div>
""", unsafe_allow_html=True) 