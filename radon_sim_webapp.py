# radon_sim_webapp.py

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import random

# --- Constants ---
halflives = {
    'Rn222': 3.3035e5,
    'Po218': 185.88,
    'Pb214': 1608,
    'Bi214': 1194,
    'Po214': 1.643e-4,
    'Pb210': 7.0325e8
}
dconst = {k: np.log(2)/v for k, v in halflives.items()}
VOLUME = 1e-3
dt = 60  # time step in seconds

# --- Decay Function Hybrid ---
def ProgenyDecay(thalf, Ni, dt, method='binomial'):
    tau = thalf / np.log(2)
    p = 1 / tau * dt
    if method == 'atom':
        N0 = Ni
        for _ in range(Ni):
            if random.random() < p:
                Ni -= 1
        return Ni, N0 - Ni
    else:
        p = min(1.0, max(0.0, p))
        decays = np.random.binomial(Ni, p)
        return Ni - decays, decays

def RAD7_CONC_TO_N(ConcBq, decay_const):
    return int(VOLUME * ConcBq / decay_const)

# --- Streamlit UI ---
st.set_page_config(page_title="DURRIDGE Radon Measurement Simulator", layout="wide")
st.title("DURRIDGE Radon Measurement Simulator - Alpha Version")

st.sidebar.header("Radon Sample Measured")
Rn222_CONC = st.sidebar.number_input("Rn 222 (Bq/m³)", min_value=0, value=200)
source = st.sidebar.radio("Constant Radon Source", ["On", "Off"]) == "On"

st.sidebar.markdown("---")
st.sidebar.subheader("Reference Levels")
st.sidebar.markdown("""
- 🌳 **Outdoors:** ~10 Bq/m³  
- ⚠️ **EPA Action Level:** ~148 Bq/m³ (≈4 pCi/L)  
- ⛏️ **Uranium Mines:** >10,000 Bq/m³  
- 🏠 **Stanley Watras’ Basement:** >100,000 Bq/m³  
- 🧪 **Dark Matter Labs:** <0.001 Bq/m³  
""")


st.sidebar.header("RAD Measurement Protocol")
cycle_time = st.sidebar.number_input("Cycle Time (min)", min_value=1, value=15) * 60
simtime = st.sidebar.number_input("Measurement Time (min)", min_value=1, value=180) * 60
mode = st.sidebar.radio("Mode", ["Sniff", "Normal"])



st.sidebar.header("Simulated Progeny Counts")
show_po218 = st.sidebar.checkbox("Show Po218", value=False)
show_po214 = st.sidebar.checkbox("Show Po214", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("**Author:** Robert Renz Marcelo Gregorio  \n**Email:** rob@durridge.co.uk  \n**Year:** 2025 Version: Alpha")


# --- Simulation ---
Rn222 = RAD7_CONC_TO_N(Rn222_CONC, dconst['Rn222'])
Po218 = Pb214 = Bi214 = Po214 = Pb210 = 0
dNsim = np.empty((0, 7), dtype=float)
po_cycle_table = []
po_conc_log = []
po218_cycle = 0
po214_cycle = 0

normal_sens = 0.014
sniff_sens = 0.0068
fudge_factor = 8.7

for time in range(0, int(simtime), dt):
    Rn222_red, Po218_gen = ProgenyDecay(halflives['Rn222'], Rn222, dt)
    Po218_red, Pb214_gen = ProgenyDecay(halflives['Po218'], Po218, dt)
    Pb214_red, Bi214_gen = ProgenyDecay(halflives['Pb214'], Pb214, dt)
    Bi214_red, Po214_gen = ProgenyDecay(halflives['Bi214'], Bi214, dt)
    Po214_red, Pb210_gen = ProgenyDecay(halflives['Po214'], Po214, dt)
    Pb210_red, _ = ProgenyDecay(halflives['Pb210'], Pb210, dt)

    po218_cycle += Po218_gen
    po214_cycle += Po214_gen

    dNsimdt = np.array([time, Po218_gen / dt, Pb214_gen / dt, Bi214_gen / dt, Po214_gen / dt, Pb210_gen / dt, 0])
    dNsim = np.vstack((dNsim, dNsimdt))
    po_conc_log.append([time, Po218, Po214])

    Rn222 = Rn222 if source else Rn222_red
    Po218 = Po218_gen + Po218_red
    Pb214 = Pb214_gen + Pb214_red
    Bi214 = Bi214_gen + Bi214_red
    Po214 = Po214_gen + Po214_red
    Pb210 = Pb210_gen + Pb210_red

    if time % cycle_time == 0 and time > 0:
        mins = time // 60
        cpm_po218 = po218_cycle / (cycle_time / 60)
        cpm_po214 = po214_cycle / (cycle_time / 60)
        po_cycle_table.append({
            'Cycle Time (min)': mins,
            'Po218 CPM': cpm_po218,
            'Po214 CPM': cpm_po214,
            'Radon Normal': (cpm_po218 + cpm_po214) / (normal_sens * fudge_factor),
            'Radon Sniff': cpm_po218 / (sniff_sens * fudge_factor),
            'Radon Normal ±2σ': 2 * (1 + np.sqrt(po218_cycle + po214_cycle + 1)) / (normal_sens * fudge_factor * (cycle_time / 60)),
            'Radon Sniff ±2σ': 2 * (1 + np.sqrt(po218_cycle + 1)) / ((sniff_sens * fudge_factor) * (cycle_time / 60))
        })
        po218_cycle = 0
        po214_cycle = 0

df = pd.DataFrame(dNsim, columns=['time', 'Po218', 'Pb214', 'Bi214', 'Po214', 'Pb210', 'NA'])
df['time'] /= 60
po_df = pd.DataFrame(po_cycle_table)
po_conc_df = pd.DataFrame(po_conc_log, columns=['time', 'Po218', 'Po214'])
po_conc_df['time'] /= 60

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 5))
if show_po218:
    ax.plot(po_conc_df['time'], po_conc_df['Po218'] * dconst['Po218'] / VOLUME,
            label='Po218', linestyle=':', marker='.', markersize=3, alpha=0.6, color='#D62728')
if show_po214:
    ax.plot(df['time'], df['Po214'] / VOLUME,
            label='Po214', linestyle=':', marker='.', markersize=3, alpha=0.6, color='#1F77B4')

if mode == 'Normal':
    y = po_df['Radon Normal']
    yerr = po_df['Radon Normal ±2σ']
elif mode == 'Sniff':
    y = po_df['Radon Sniff']
    yerr = po_df['Radon Sniff ±2σ']
else:
    y = yerr = None

x = po_df['Cycle Time (min)']
ax.errorbar(x, y, yerr=yerr, fmt='-o', color='black', ecolor='blue', elinewidth=1, capsize=4,
            label=f'Radon {mode} Mode')

ax.set_xlabel('Time (Minutes)')
ax.set_ylabel('Concentration (Bq/m³)')
ax.grid(True)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.legend()
st.pyplot(fig)

