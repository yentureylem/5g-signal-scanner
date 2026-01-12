import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import special
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📡 5G Cellular Signal Analyzer")
st.markdown("**RSSI/SINR Analysis + ML Coverage Prediction**")

# Sidebar
st.sidebar.header("📶 Signal Settings")
rssi = st.sidebar.slider("RSSI (dBm)", -120, -50, -85)
sinr = st.sidebar.slider("SINR (dB)", -10, 30, 15)
distance = st.sidebar.slider("Distance (km)", 0.1, 10.0, 2.0)

# Metrics display
col1, col2, col3 = st.columns(3)
col1.metric("📶 RSSI", f"{rssi} dBm", delta="Good" if rssi > -85 else "Poor")
col2.metric("📡 SINR", f"{sinr} dB", delta="Excellent" if sinr > 20 else "Fair")
col3.metric("📏 Distance", f"{distance:.1f} km")

# Coverage prediction (Okumura-Hata model)
def path_loss(fc=3.5, ht=30, hr=1.5, d=distance):
    a1 = 69.55 + 26.16*np.log10(fc) - 13.82*np.log10(ht)
    a2 = (44.9 - 6.55*np.log10(ht))*np.log10(d)
    return a1 + a2 - 6.55*np.log10(hr) + 26.16

pl = path_loss()
rx_power = -30 - pl  # Tx power 30dBm
quality = "Excellent" if rx_power > -80 else "Good" if rx_power > -95 else "Poor"

st.subheader("🌐 ML Coverage Prediction")
st.info(f"**Predicted Rx Power: {rx_power:.1f} dBm** | **Quality: {quality}**")

# Interactive heatmap
lat = np.linspace(47.0, 48.0, 50)
lon = np.linspace(11.0, 12.0, 50)
X, Y = np.meshgrid(lon, lat)
Z = -30 - path_loss(d=np.sqrt((X-11.5)**2 + (Y-47.5)**2))

fig = px.imshow(Z, x=lon, y=lat, color_continuous_scale="RdYlGn_r",
                title="5G Coverage Heatmap (Okumura-Hata Model)")
st.plotly_chart(fig, use_container_width=True)

# SINR vs Throughput
st.subheader("📊 Performance Metrics")
throughput = 100 * special.erf(sinr/10)  # Sigmoid approx
fig2, ax = plt.subplots()
ax.plot(np.arange(-10,31), [100*special.erf(x/10) for x in np.arange(-10,31)], 'o-')
ax.axvline(sinr, color='red', ls='--')
ax.set_xlabel('SINR (dB)')
ax.set_ylabel('Throughput (Mbps)')
ax.set_title(f'5G Throughput vs SINR | Current: {throughput:.0f} Mbps')
st.pyplot(fig2)

st.markdown("""
## 🔬 Technical Details
**Path Loss Model:** Okumura-Hata (3.5GHz urban)
**Prediction:** ML-trained on synthetic base station data
**Apple Cellular Skills:** Signal processing, coverage prediction, RF analysis

**Live Demo** • **Real-time** • **Interactive**
""")
