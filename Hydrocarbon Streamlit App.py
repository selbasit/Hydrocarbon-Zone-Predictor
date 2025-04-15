# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Hydrocarbon Zone Predictor", layout="wide")
st.title("🛢️ Hydrocarbon Zone Predictor from Well Logs")

# Load model
@st.cache_resource
def load_model_cached():
    return load_model("hydrocarbon_model.h5")

model = load_model_cached()

# Upload CSV
uploaded_file = st.file_uploader("Upload Well Log CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data Preview")
    st.write(df.head())

    # Ensure required columns exist
    required_cols = ['GR', 'RD', 'RS', 'CNC', 'ZDEN']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
    else:
        # Preprocessing
        df.replace(-999.25, np.nan, inplace=True)
        df.dropna(subset=required_cols, inplace=True)

        # Feature engineering
        df['Resistivity_ratio'] = df['RD'] / df['RS']
        df['Porosity_density'] = (2.65 - df['ZDEN']) / (2.65 - 1.0)
        df['ND_diff'] = df['CNC'] - df['Porosity_density']

        features = df[['GR', 'RD', 'RS', 'CNC', 'ZDEN', 'Resistivity_ratio', 'ND_diff']]

        # Generate windowed sequences (reshape to 3D if model expects 3D input)
        window_size = 10  # Example: use 10 consecutive samples per prediction
        sequences = []
        for i in range(len(features) - window_size + 1):
            window = features.iloc[i:i+window_size].values
            sequences.append(window)
        X_seq = np.array(sequences)

        # Scale per feature across all sequences
        X_reshaped = X_seq.reshape(-1, X_seq.shape[-1])
        scaler = StandardScaler()
        X_scaled_flat = scaler.fit_transform(X_reshaped)
        X_scaled_seq = X_scaled_flat.reshape(X_seq.shape)

        # Predict
        y_pred_prob = model.predict(X_scaled_seq)

        # Pad results to align with original dataframe
        pad_front = window_size // 2
        pad_back = len(df) - len(y_pred_prob) - pad_front
        y_padded = np.pad(y_pred_prob.squeeze(), (pad_front, pad_back), mode='edge')

        df['Hydrocarbon_Prob'] = y_padded
        threshold = st.slider("Prediction Threshold", 0.0, 1.0, 0.5, 0.01)
        df['Prediction'] = (df['Hydrocarbon_Prob'] >= threshold).astype(int)

        # Show prediction
        st.subheader("Prediction Results")
        st.write(df[['DEPT', 'Hydrocarbon_Prob', 'Prediction']].head())

        # Visualization: Log curves and probability curve
        st.subheader("Log Curves and Hydrocarbon Probability")
        fig, axs = plt.subplots(1, 4, figsize=(20, 10), sharey=True)

        axs[0].plot(df['GR'], df['DEPT'], label='GR', color='green')
        axs[0].set_xlabel('GR')
        axs[0].invert_yaxis()
        axs[0].grid()

        axs[1].plot(df['RD'], df['DEPT'], label='RD', color='red')
        axs[1].set_xlabel('RD')
        axs[1].grid()

        axs[2].plot(df['CNC'], df['DEPT'], label='CNC', color='purple')
        axs[2].set_xlabel('CNC')
        axs[2].grid()

        axs[3].plot(df['Hydrocarbon_Prob'], df['DEPT'], label='HC Probability', color='blue')
        axs[3].set_xlabel('HC Probability')
        axs[3].grid()

        for ax in axs:
            ax.legend()

        st.pyplot(fig)

        # Download option
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results CSV", csv, "prediction_results.csv", "text/csv")
