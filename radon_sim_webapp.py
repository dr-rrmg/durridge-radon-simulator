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

dconst = {k: np.log(2) / v for k, v in halflives.items()}
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


# --- Sidebar Sections ---
st.sidebar.markdown("**DURRIDGE Radon Measurement Simulator (Experimental Code Ver.)**")

with st.sidebar.expander("⚛️ Radon Sample", expanded=True):
    Rn222_CONC = st.number_input("Rn 222 (Bq/m³)", min_value=0, value=200)
    source = st.radio("Constant Source", ["On", "Off"]) == "On"

with st.sidebar.expander("🎯 Measurement Protocol", expanded=True):
    protocols = {
        "Sniff": {"cycle": 5, "time": 180, "mode": "Sniff"},
        "1-day": {"cycle": 30, "time": 1440, "mode": "Auto"},
        "2-day": {"cycle": 60, "time": 2880, "mode": "Auto"},
        "Weeks": {"cycle": 120, "time": 10080, "mode": "Auto"}
    }

    selected_preset = st.selectbox("Select Protocol", list(protocols.keys()))
    preset = protocols[selected_preset]

    use_custom = st.checkbox("✏️ Customise protocol manually")

    if use_custom:
        cycle_time = st.number_input("Cycle Time (min)", min_value=1, value=preset["cycle"]) * 60
        simtime = st.number_input("Measurement Duration (min)", min_value=1, value=preset["time"]) * 60
        mode = st.radio(
            "Mode",
            ["Sniff", "Normal", "Auto"],
            index=["Sniff", "Normal", "Auto"].index(preset["mode"])
        )
    else:
        cycle_time = preset["cycle"] * 60
        simtime = preset["time"] * 60
        mode = preset["mode"]

    st.markdown(f"""
    **Cycle Time:** {cycle_time // 60:.0f} min  
    **Duration:** {simtime // 60:.0f} min  
    **Mode:** {mode}
    """)

with st.sidebar.expander("📚 Reference Levels", expanded=True):
    st.markdown("""
    - 🌳 **Outdoors:** ~10 Bq/m³  
    - ⚠️ **EPA Action Level:** ~148 Bq/m³  
    - ⛏️ **Uranium Mines:** >10,000 Bq/m³  
    - 🏠 **Stanley Watras’ Basement:** >100,000 Bq/m³  
    - 🧪 **Dark Matter Labs:** <0.001 Bq/m³  
    """)

with st.sidebar.expander("📟 Display Options"):
    show_po218 = st.checkbox("Show Po218", value=False)
    show_po214 = st.checkbox("Show Po214", value=False)

    bar_mode = st.selectbox(
        "Window Bar Chart",
        ["Latest Cycle CPM", "Total Measurement"]
    )

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Author:** Robert Renz Marcelo Gregorio  \n"
    "**Email:** rob@durridge.co.uk  \n"
    "**Year:** 2025  \n"
    "**Version:** Experimental Code"
)


# --- Simulation ---
Rn222 = RAD7_CONC_TO_N(Rn222_CONC, dconst['Rn222'])
Po218 = Pb214 = Bi214 = Po214 = Pb210 = 0

dNsim = np.empty((0, 7), dtype=float)
po_cycle_table = []
po_conc_log = []

po218_cycle = 0
po214_cycle = 0

total_po218_counts = 0
total_po214_counts = 0

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

    total_po218_counts += Po218_gen
    total_po214_counts += Po214_gen

    dNsimdt = np.array([
        time,
        Po218_gen / dt,
        Pb214_gen / dt,
        Bi214_gen / dt,
        Po214_gen / dt,
        Pb210_gen / dt,
        0
    ])

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

        radon_sniff = cpm_po218 / (sniff_sens * fudge_factor)
        radon_sniff_err = 2 * (1 + np.sqrt(po218_cycle + 1)) / (
            (sniff_sens * fudge_factor) * (cycle_time / 60)
        )

        radon_normal = (cpm_po218 + cpm_po214) / (normal_sens * fudge_factor)
        radon_normal_err = 2 * (1 + np.sqrt(po218_cycle + po214_cycle + 1)) / (
            normal_sens * fudge_factor * (cycle_time / 60)
        )

        po_cycle_table.append({
            'Cycle Time (min)': mins,
            'Po218 CPM': cpm_po218,
            'Po214 CPM': cpm_po214,
            'Po218 Counts': po218_cycle,
            'Po214 Counts': po214_cycle,
            'Radon Normal': radon_normal,
            'Radon Sniff': radon_sniff,
            'Radon Normal ±2σ': radon_normal_err,
            'Radon Sniff ±2σ': radon_sniff_err,
            'Radon Auto': radon_sniff if time <= 10800 else radon_normal,
            'Radon Auto ±2σ': radon_sniff_err if time <= 10800 else radon_normal_err
        })

        po218_cycle = 0
        po214_cycle = 0


# --- DataFrames ---
df = pd.DataFrame(
    dNsim,
    columns=['time', 'Po218', 'Pb214', 'Bi214', 'Po214', 'Pb210', 'NA']
)

df['time'] /= 60

po_df = pd.DataFrame(po_cycle_table)

po_conc_df = pd.DataFrame(
    po_conc_log,
    columns=['time', 'Po218', 'Po214']
)

po_conc_df['time'] /= 60


# --- Main RAD7 Plot ---
fig, ax = plt.subplots()

if show_po218:
    ax.plot(
        po_conc_df['time'],
        po_conc_df['Po218'] * dconst['Po218'] / VOLUME,
        label='Po218',
        linestyle=':',
        marker='.',
        markersize=3,
        alpha=0.6,
        color='#D62728'
    )

if show_po214:
    ax.plot(
        po_conc_df['time'],
        po_conc_df['Po214'] * dconst['Po214'] / VOLUME,
        label='Po214',
        linestyle=':',
        marker='.',
        markersize=3,
        alpha=0.6,
        color='#1F77B4'
    )

if not po_df.empty:
    if mode == 'Normal':
        y = po_df['Radon Normal']
        yerr = po_df['Radon Normal ±2σ']
    elif mode == 'Sniff':
        y = po_df['Radon Sniff']
        yerr = po_df['Radon Sniff ±2σ']
    elif mode == 'Auto':
        y = po_df['Radon Auto']
        yerr = po_df['Radon Auto ±2σ']
    else:
        y = yerr = None

    x = po_df['Cycle Time (min)']

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt='-o',
        color='black',
        ecolor='blue',
        elinewidth=1,
        capsize=4,
        label=f'Radon {mode} Mode'
    )

ax.set_xlabel('Time (Minutes)')
ax.set_ylabel('Concentration (Bq/m³)')
ax.grid(True)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.legend()

st.pyplot(fig, use_container_width=True)


# --- RAD7 Window Bar Chart ---
st.subheader("RAD7 Window Bar Chart")

if not po_df.empty:
    if bar_mode == "Latest Cycle CPM":
        latest_cycle = po_df.iloc[-1]

        cpm_a = latest_cycle['Po218 CPM']
        cpm_b = 0
        cpm_c = latest_cycle['Po214 CPM']
        cpm_d = 0



    else:
        total_minutes = simtime / 60

        cpm_a = total_po218_counts / total_minutes
        cpm_b = 0
        cpm_c = total_po214_counts / total_minutes
        cpm_d = 0



    windows = {
        "A": {"x1": 5.5, "x2": 6.4, "cpm": cpm_a, "label": r"$^{218}$Po"},
        "B": {"x1": 6.8, "x2": 7.2, "cpm": cpm_b, "label": r"$^{216}$Po"},
        "C": {"x1": 7.6, "x2": 8.3, "cpm": cpm_c, "label": r"$^{214}$Po"},
        "D": {"x1": 8.6, "x2": 9.0, "cpm": cpm_d, "label": r"$^{212}$Po"},
    }

    fig_bar, ax_bar = plt.subplots(figsize=(8, 4))

    # Background vertical stripes
    for xstripe in np.arange(4.0, 9.3, 0.1):
        ax_bar.axvspan(xstripe, xstripe + 0.04, color="lightgrey", alpha=0.5, linewidth=0)

    # Window regions
    for name, w in windows.items():
        colour = "black" if w["cpm"] > 0 else "0.7"
        ax_bar.axvspan(w["x1"], w["x2"], ymin=0, ymax=0.82, color=colour)

        xmid = (w["x1"] + w["x2"]) / 2
        ax_bar.text(xmid, 0.88, name, ha="center", va="center",
                    fontsize=16, fontweight="bold", transform=ax_bar.get_xaxis_transform())

        ax_bar.text(xmid, 1.03, w["label"], ha="center", va="bottom",
                    fontsize=14, transform=ax_bar.get_xaxis_transform())

        ax_bar.text(xmid, -0.13, f"{w['cpm']:.3g}", ha="center", va="top",
                    fontsize=12, transform=ax_bar.get_xaxis_transform())

    # Extra isotope label at left
    ax_bar.text(4.7, 1.03, r"$^{210}$Po", ha="center", va="bottom",
                fontsize=14, transform=ax_bar.get_xaxis_transform())

    ax_bar.text(4.25, -0.13, "CPM:", ha="right", va="top",
                fontsize=12, transform=ax_bar.get_xaxis_transform())


    ax_bar.set_xlim(4.2, 9.3)
    ax_bar.set_ylim(0, 1)

    ax_bar.set_yticks([])

    ax_bar.set_xticks([5, 6, 7, 8, 9])
    ax_bar.set_xticklabels(["5MeV", "6MeV", "7MeV", "8MeV", "9MeV"])

    ax_bar.tick_params(axis='x', length=8)
    ax_bar.grid(False)

    st.pyplot(fig_bar, use_container_width=True)

else:
    st.warning("No completed cycles yet. Increase the measurement duration or reduce the cycle time.")
