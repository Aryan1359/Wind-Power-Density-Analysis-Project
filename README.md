# Module 5 — Wind Energy Assignment

## Wind Power Density Analysis (California & Offshore Regions)

**Course:** A First Course in Renewable Energy
**Program:** UC Berkeley – MASE
**Term:** Spring 2026

<div align="center">

  <img src="https://lh3.googleusercontent.com/sitesv/APaQ0SRCFUx2M6M4WkLO4-qPhwsXtog3TNv-y_d5BYBQhAWEmx9IHBHgbOYryYGdeIFPfy4fi0Yu8vG6u5vMEqr1z5HFNHBjX9vuRci0OehvfoHcEhH3_3Q_wv5OH_aK9D0tWUikRCTTvrRdTTjaN2W1bdfr8nDQ7fYo36TofbnCHvh7hUkJW2pGSk_cTmnPz7KxAtriWSkKb73pwJ8MVxyJRBM_JC6RQNlruniN1KQ=w1280" alt="Wind power density visualization 1" width="49%" />
  <img src="https://lh3.googleusercontent.com/sitesv/APaQ0STRMzOs5yP07CgMNn35sT0bWmI9ez_QSYOZahkm69E3ZPnFucPLN3a0oQImgLpBnSsA1OlhVAknsRbur6Yev5Du9A5uGqCGByZwqy6p6sbGhnxIkke1NlmNuJ8lIF33WsoJ4TmbIGvAOyygJ6m0Oav_ahyqu0AB-R33eisZ1xJHxrpZHyK-gQ0m9Zq6c4plVXhDGkKaLWTzAJB2ODnB7zj7MeiV3fGbYueioUc=w1280" alt="Wind power density visualization 2" width="49%" />

</div>

---

### 📌 Assignment Context

**Graded Project — Module 5**

This repository contains the **complete, reproducible analysis** for the
**Wind Power Density Analysis Project**, focused on evaluating wind energy
potential across **California and offshore Pacific regions** using real-world
wind velocity data from **Zoom Earth**.

The work follows a **research-oriented engineering workflow**, combining:

* Scientific data extraction
* Physics-based energy calculations
* Professional visualization
* Full reproducibility via code and environment control

---

### 👤 Author & Metadata

* **Student:** Aryan Yaghobi
* **Module:** 5 – Wind Energy
* **Assignment:** Wind Power Density Analysis Project
* **Submission Type:** Code + Visualization + PDF Report

---

### 🔗 Project Navigation

* 📄 **Final Report (PDF):** Submitted via course platform
* 🌐 **Step-by-step Methodology (Google Site):**
  [https://sites.google.com/view/renewable-energy2026/module-5](https://sites.google.com/view/renewable-energy2026/module-5)
* 💻 **Source Code & Reproducibility:** This GitHub repository

---

> **Note:**
> This repository serves as the **canonical source of truth** for all code,
> environment setup, and generated figures.
> The Google Site documents the *process and interpretation*;
> GitHub ensures *technical transparency and reproducibility*.

---

## 📖 Project Overview

This project analyzes **wind power density** across the state of California and its offshore Pacific regions using real-world wind velocity data from **Zoom Earth**.
The objective is to convert a visual wind map into **quantitative, grid-based wind speed and power density data**, following standard engineering and scientific analysis practices.

The workflow emphasizes:

* Reproducibility
* Transparent methodology
* Scientific visualization
* Professional engineering documentation

---

## 🎯 Objectives

* Extract wind speed data from a color-coded satellite wind map
* Convert visual data into numerical wind speed values
* Compute wind power density using physical principles
* Compare **onshore vs offshore** wind energy potential
* Produce publication-quality visualizations

---

## 🗺️ Data Source

* **Zoom Earth** ([https://zoom.earth](https://zoom.earth))
* Wind velocity layer with color legend (mph)
* Single screenshot including:

  * Entire state of California
  * Offshore Pacific Ocean
  * Wind speed color scale

---

## 🧠 Methodology

### 1. Image Grid Processing

* The Zoom Earth screenshot is divided into a **20 × 20 grid**
* Each grid cell represents an equal spatial area
* Average **RGB color values** are computed per cell

### 2. Wind Speed Mapping

* RGB values are mapped to wind speeds (mph) using the Zoom Earth legend
* Wind speeds are converted to SI units:

[
v_{m/s} = v_{mph} \times 0.44704
]

### 3. Wind Power Density Calculation

Wind power density is computed for each grid cell using:

[
P = \frac{1}{2} \rho v^3 / 1000
]

Where:

* ( \rho = 1.225 , \text{kg/m}^3 ) (air density)
* ( v ) = wind speed in m/s
* Output units: **kW/m²**

### 4. Visualization

A four-panel scientific figure is generated:

1. Original Zoom Earth image
2. 20×20 averaged-color grid
3. Wind speed map (m/s)
4. Wind power density map (kW/m²)

Each panel maintains:

* Correct aspect ratio
* Grid lines
* Colorbars
* Numerical annotations

---

## 📊 Outputs

* Four-panel visualization (PNG)
* Grid-based wind speed map
* Grid-based wind power density map
* Summary statistics:

  * Minimum
  * Maximum
  * Mean values
* Onshore vs offshore comparison

---

## 🧰 Tools & Technologies

* **Python 3.11**
* NumPy (numerical computation)
* Matplotlib (scientific plotting)
* Pillow / OpenCV (image processing)
* Anaconda (environment management)
* Git & GitHub (version control)

---

## ⚙️ Environment Setup (Reproducibility)

### 1. Create the environment

```bash
conda create -n wind_m5 python=3.11
conda activate wind_m5
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

1. Activate the environment:

```bash
conda activate wind_m5
```

2. Place the Zoom Earth screenshot in:

```
data/raw/
```

3. Run the main analysis script:

```bash
python src/main_analysis.py
```

4. Outputs (figures and statistics) will be saved to:

```
outputs/
```

---

## ⚠️ Assumptions & Limitations

* Wind speeds are estimated via color-to-value mapping and are approximate
* Spatial resolution is limited by the 20 × 20 grid
* Results are intended for comparative and educational analysis
* This is not a site-specific engineering feasibility study

---

## 📄 Academic Context

This project was completed as part of **Module 5 – Wind Energy** in *A First Course in Renewable Energy*.
The workflow reflects standard practices used in wind resource assessment, environmental data analysis, and research-oriented engineering projects.
