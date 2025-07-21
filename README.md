
# Variable Classification Streamlit App

This repository contains a **Streamlit application** for classifying variables based on their properties, generating reports, and visualizations.

## Features

- **Robust Data Loading**: Supports loading CSV and Excel files with automatic delimiter and header detection.
- **Statistical Feature Calculation**: Computes key statistical metrics for each numerical variable.
- **Rule-Based Categorization**: Classifies variables into categories (e.g., constant, static, boolean, counter, unstable, periodic, variable, sensor value, non-numeric) based on predefined rules.
- **K-Means Clustering**: Groups variables with similar characteristics.
- **Intelligent Recommendations**: Provides recommendations for each variable (e.g., highly relevant, review, better ignore, redundant (highly correlated)).
- **Visualizations**: Includes interactive charts (category bar charts, Sunburst charts, correlation heatmaps, time series plots) for better data understanding.
- **Report Generation**: Creates a comprehensive data profile report using `ydata-profiling`.

---

##  Setup

To run this application locally, follow these steps:

### 1. Clone the GitHub Repository
```bash
git clone https://github.com/<your_github_username>/variable_classifier_app.git
cd variable_classifier_app
```

### 2. Create and Activate a Virtual Environment (Recommended)
```bash
python -m venv venv
# For Windows:
venv\Scripts\activate
# For macOS/Linux:
source venv/bin/activate
```

### 3. Install Required Dependencies
Ensure you have the `requirements.txt` file in your project:
```bash
pip install -r requirements.txt
```

---

##  Running the Streamlit Application

```bash
streamlit run app6.py
```

This command will open the application in your web browser.

---

## Classification and Clustering Logic

The logic for classifying and clustering variables is based on a **hybrid approach** combining rule-based categorization and K-Means clustering.

### Diagram:

<img width="3840" height="2986" alt="image" src="https://github.com/user-attachments/assets/2735f970-396e-4ec3-9380-d8df86c48a1d" />

---

## Usage

1. **Upload Data**: Use "Step 1: Upload Data" to upload your CSV or Excel file.
2. **Start Analysis**: Click "Start Analysis" in "Step 2: Perform Analysis and Classification". This will calculate features, classify variables, and perform clustering.
3. **View Results**: Explore tables and visualizations in "Step 3: Visualize Results".
4. **Download Reports**: Use "Step 4" and "Step 5" to export final data and a full data profile.

---

##  Project Structure
```
variable_classifier_app/
├── output/
│   ├── data_report.html
│   ├── Data profile.html
│   └── cluster_visualization.png
├── data/
├── .git/
├── requirements.txt
├── README.md
├── app6.py
└── .gitignore
```

---

## 📬 Contact
Feel free to contribute or ask questions via GitHub Issues or Pull Requests.
