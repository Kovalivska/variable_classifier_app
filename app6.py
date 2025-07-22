import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.spatial.distance import cdist
import csv
import io
import os
import plotly_express as px
# import sweetviz as sv # Removed due to Python 3.13 incompatibility
import sklearn

# Suppress warnings for a cleaner output
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Configuration (MUST be the first Streamlit command) ---
st.set_page_config(layout="wide", page_title="Variable Classification App")

# --- Library Installation and Version Check for Colab Synchronization ---
# These commands will attempt to install/update the specified libraries.
# For production environments, it's recommended to manage dependencies via requirements.txt
# or before starting Streamlit, but for consistency here.
try:
    # Check current versions
    import pandas as pd_check
    import numpy as np_check
    import sklearn as sk_check
    
    required_pandas = '2.2.3'
    required_numpy = '1.26.4'
    required_sklearn = '1.6.1' # Version that was in the previous Colab installation

    versions_match = (
        pd_check.__version__ == required_pandas and
        np_check.__version__ == required_numpy and
        sk_check.__version__ == required_sklearn
    )

    if not versions_match:
        # Print to console/log, as st.warning cannot be called before set_page_config
        print(f"Library versions do not match Colab. Attempting to install specific versions: "
              f"pandas=={required_pandas}, numpy=={required_numpy}, scikit-learn=={required_sklearn}")
        # Use os.system to execute pip install
        # Removed sweetviz from installation command as it's causing issues
        os.system(f'pip install pandas=={required_pandas} numpy=={required_numpy} scikit-learn=={required_sklearn} plotly_express openpyxl --upgrade --quiet')
        st.experimental_rerun() # Rerun the app after installation
    else:
        # Display success message only after set_page_config has run
        st.success(f"Library versions match Colab: pandas {pd_check.__version__}, numpy {np_check.__version__}, scikit-learn {sk_check.__version__}")

except ImportError:
    # Print to console/log, as st.warning cannot be called before set_page_config
    print("Required libraries not found. Attempting installation.")
    # Removed sweetviz from installation command as it's causing issues
    os.system('pip install pandas==2.2.3 numpy==1.26.4 scikit-learn==1.6.1 plotly_express openpyxl --upgrade --quiet')
    st.experimental_rerun() # Rerun the app after installation

# --- Helper Functions ---

def load_data_robustly(uploaded_file):
    """
    Automatic loading of CSV or Excel files with structure checking.
    Directly adapted from the user's script and adjusted for Streamlit.
    """
    df = None
    file_extension = os.path.splitext(uploaded_file.name)[-1].lower()

    if file_extension in ['.xls', '.xlsx', '.xlsm']:
        st.info(f"Excel file detected: {uploaded_file.name}")
        try:
            # Read as bytes and then use io.BytesIO
            xls = pd.read_excel(io.BytesIO(uploaded_file.getvalue()), sheet_name=None, decimal=",")
            st.info(f"Available Sheets: {list(xls.keys())}")
            df = next(iter(xls.values()))
            st.success("Excel data successfully loaded.")
        except Exception as e:
            st.error(f"Error loading Excel file: {str(e)}")
    else:
        st.info(f"Attempting to load CSV file: {uploaded_file.name}")
        common_delimiters = [',', ';', '\t', '|']
        file_content = uploaded_file.getvalue().decode('utf-8') # Decode file content

        for delimiter in common_delimiters:
            try:
                # Try header=0 first
                df_attempt_header0 = pd.read_csv(
                    io.StringIO(file_content), # Use StringIO for string content
                    delimiter=delimiter,
                    low_memory=False,
                    header=0,
                    encoding='utf-8',
                    decimal=","
                )
                if len(df_attempt_header0.columns) > 1 and df_attempt_header0.shape[0] > 0:
                    # Check for a reasonable number of non-empty columns after dropping all-NaN columns
                    if df_attempt_header0.dropna(axis=1, how='all').shape[1] > 5:
                        df = df_attempt_header0
                        st.success(f"Successfully loaded with delimiter '{delimiter}' and header=0.")
                        break
            except Exception:
                pass

            if df is None: # If header=0 failed, try header=1
                try:
                    df_attempt_header1 = pd.read_csv(
                        io.StringIO(file_content),
                        delimiter=delimiter,
                        low_memory=False,
                        header=1,
                        encoding='utf-8',
                        decimal=","
                    )
                    if len(df_attempt_header1.columns) > 1 and df_attempt_header1.shape[0] > 0:
                        # Check for a reasonable number of non-empty columns after dropping all-NaN columns
                        if df_attempt_header1.dropna(axis=1, how='all').shape[1] > 5:
                            df = df_attempt_header1
                            st.success(f"Successfully loaded with delimiter '{delimiter}' and header=1.")
                            break
                except Exception:
                    pass

    if df is None:
        st.error("Error: File could not be loaded. Please try another file or check its format/encoding.")
    else:
        st.write("Dataset successfully loaded.")
        st.write("Shape:", df.shape)
        st.write("First 5 rows:")
        st.dataframe(df.head())
        
    return df

def calculate_features(df_clean_numeric):
    """
    Calculates statistical features for each numeric variable.
    Directly adapted from the user's script, with MAD calculation fixed.
    """
    features = []
    for col in df_clean_numeric.columns:
        if pd.api.types.is_numeric_dtype(df_clean_numeric[col]):
            series = df_clean_numeric[col].dropna()
            if series.empty:
                features.append({
                    'Name': col,
                    'count': 0, 'mean': np.nan, 'std': np.nan, 'min': np.nan,
                    '25%': np.nan, '50%': np.nan, '75%': np.nan, 'max': np.nan,
                    'range': np.nan, 'variance': np.nan, 'median': np.nan,
                    'mad': np.nan, 'skewness': np.nan, 'kurtosis': np.nan,
                    'unique_values': 0, 'num_changes': 0, 'pct_changes': np.nan,
                    'pct_zeros': np.nan, 'pct_nans': 1.0, 'abs_max': np.nan,
                    'change_frequency': np.nan, 'max_consecutive_same': np.nan,
                    'min_consecutive_same': np.nan, 'Perzentilspanne': np.nan,
                    'Extremwertanteil': np.nan
                })
                continue

            # Basic Statistics
            count = series.count()
            mean = series.mean()
            std = series.std()
            _min = series.min()
            _25 = series.quantile(0.25)
            _50 = series.quantile(0.50)
            _75 = series.quantile(0.75)
            _max = series.max()

            # Additional Metrics
            data_range = _max - _min
            variance = series.var()
            median = series.median()
            # Fixed MAD calculation
            mad = (series - series.mean()).abs().mean()
            skewness = series.skew()
            kurtosis = series.kurtosis()
            unique_values = series.nunique()

            # Changes and Frequency
            if count > 1:
                # Corrected IndexError: ensure boolean index matches indexed array
                change_indices = series.diff().dropna()[series.diff().dropna() != 0].index.tolist()
                num_changes = len(change_indices)
                pct_changes = num_changes / (count - 1)
                
                if num_changes > 0:
                    change_frequency = np.mean(np.diff(change_indices))
                else:
                    change_frequency = np.nan
            else:
                num_changes = 0
                pct_changes = 0
                change_frequency = np.nan

            # Percentage of Zeros
            pct_zeros = (series == 0).sum() / count if count > 0 else 0

            # Percentage of NaN values (calculated from original column)
            pct_nans = df_clean_numeric[col].isnull().sum() / len(df_clean_numeric[col])

            # Absolute Max Value
            abs_max = series.abs().max()

            # Max and Min number of consecutive same values
            rle = []
            if count > 0:
                current_val = series.iloc[0]
                current_len = 0
                for val in series:
                    if val == current_val:
                        current_len += 1
                    else:
                        rle.append(current_len)
                        current_val = val
                        current_len = 1
                rle.append(current_len) # Add the last run

                max_consecutive_same = max(rle) if rle else 0
                min_consecutive_same = min(rle) if rle else 0
            else:
                max_consecutive_same = 0
                min_consecutive_same = 0

            # Percentile Range (95th Percentile - 5th Percentile)
            per_95 = series.quantile(0.95)
            per_05 = series.quantile(0.05)
            Perzentilspanne = per_95 - per_05

            # Outlier Share (values outside 1.5 * IQR range)
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            extremwert_count = ((series < lower_bound) | (series > upper_bound)).sum()
            Extremwertanteil = extremwert_count / count if count > 0 else 0

            features.append({
                'Name': col,
                'count': count, 'mean': mean, 'std': std, 'min': _min,
                '25%': _25, '50%': _50, '75%': _75, 'max': _max,
                'range': data_range, 'variance': variance, 'median': median,
                'mad': mad, 'skewness': skewness, 'kurtosis': kurtosis,
                'unique_values': unique_values, 'num_changes': num_changes, 'pct_changes': pct_changes,
                'pct_zeros': pct_zeros, 'pct_nans': pct_nans, 'abs_max': abs_max,
                'change_frequency': change_frequency,
                'max_consecutive_same': max_consecutive_same,
                'min_consecutive_same': min_consecutive_same,
                'Perzentilspanne': Perzentilspanne,
                'Extremwertanteil': Extremwertanteil
            })
    return pd.DataFrame(features)

def categorize_variables(ergebnis_df_raw):
    """
    Categorizes variables based on calculated features.
    Directly adapted from the user's script.
    """
    ergebnis_df = ergebnis_df_raw.copy()

    # Get a reference to the original DataFrame from session state
    original_df = st.session_state['original_df']

    def classify(row):
        col_name = row['Name']

        # Ensure col_name exists in the original_df columns
        if col_name not in original_df.columns:
            return 'unknown/error' # Or handle as appropriate if a column name is missing

        # Rule 0: Check if original column is non-numeric, or if it's empty/all NaN in raw features
        if not pd.api.types.is_numeric_dtype(original_df[col_name]):
            return 'non-numeric'
        
        if pd.isna(row['count']) or row['count'] == 0:
            return 'non-numeric/empty' # For empty numeric columns


        # Rule 1: Constant
        if row['std'] == 0 or row['num_changes'] == 0:
            return 'constant'
        # Rule 2: Static (almost constant)
        if row['std'] < 0.01: # Threshold for "almost constant"
            return 'static'
        # Rule 3: Boolean (only 2 unique values and few changes)
        if row['unique_values'] <= 2 and row['num_changes'] <= 10: # Increased from 2 to 10 changes, as Boolean can have more than 2 changes
            return 'boolean'
        # Rule 4: Status (few unique values, but more than 2, and few changes)
        if row['unique_values'] <= 10 and row['num_changes'] <= 10: # Adjusted for Status
            return 'status'
        # Rule 5: Counter-like (min value >= 0, low number of unique values, but many changes compared to unique values)
        # Additional heuristic: 50% percentile not 0, and max value significantly larger than min value
        if row['min'] >= 0 and \
           row['unique_values'] > 10 and \
           row['pct_changes'] > 0.05 and \
           row['50%'] > 0 and \
           row['max'] / (row['min'] + 1e-6) > 2: # Max is at least twice as large as Min (to exclude "almost constant" counters)
            return 'counter'
        # Rule 6: Unstable (high outlier share)
        if row['Extremwertanteil'] > 0.20:
            return 'unstable'
        # Rule 7: Noise signal (low percentile range)
        if row['Perzentilspanne'] < 0.05 and row['pct_changes'] > 0.5: # Percentile range very small, but many changes
            return 'noise signal'
        # Rule 8: Periodic (high change frequency, but not "noisy")
        # Difficult to detect directly from features. Simplified assumption: many changes, but not "noise signal"
        if row['pct_changes'] > 0.2 and row['Perzentilspanne'] >= 0.05: # Many changes, but not too "tight"
             return 'periodic'
        # Rule 9: Variable (high spread, many unique values, many changes)
        if row['std'] > 0.01 and row['unique_values'] > 10 and row['pct_changes'] > 0.01:
            return 'variable'
        # Default for numeric values that were not specifically classified
        # This fallback applies if it's numeric but didn't fit other rules
        if pd.api.types.is_numeric_dtype(original_df[col_name]):
            return 'sensor value'
        
        return 'unknown' # Fallback for anything else, though should be caught by 'non-numeric' or 'non-numeric/empty'

    ergebnis_df['Kategorie'] = ergebnis_df.apply(classify, axis=1)

    return ergebnis_df


def find_optimal_clusters_kmeans(data_for_clustering):
    """
    Finds the optimal number of clusters using the Elbow Method.
    Directly adapted from the user's script.
    """
    if data_for_clustering.empty or data_for_clustering.shape[0] < 2:
        return 1 # Or handle as error/not applicable

    # Max clusters should not exceed number of samples
    K_range = range(1, min(11, data_for_clustering.shape[0]))
    if len(K_range) < 1:
        return 1

    inertias = []
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data_for_clustering)
        inertias.append(kmeans.inertia_)

    # Find the "elbow" point
    diffs = np.diff(inertias)
    diff_ratios = np.abs(np.diff(diffs) / diffs[:-1])
    # Find the largest "bend" where the reduction in inertia starts to diminish significantly
    if len(diff_ratios) > 0:
        optimal_k_elbow = diff_ratios.argmax() + 2 # +1 for diff, +1 for original K_range index
    else:
        optimal_k_elbow = K_range[-1] if K_range else 1 # Fallback

    return optimal_k_elbow


def apply_kmeans_clustering(ergebnis_df, optimal_k):
    """
    Performs K-Means Clustering on the normalized feature vectors.
    Directly adapted from the user's script.
    """
    # Select features for clustering
    features_for_clustering = [
        'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'range',
        'variance', 'median', 'mad', 'skewness', 'kurtosis', 'unique_values',
        'num_changes', 'pct_changes', 'pct_zeros', 'pct_nans', 'abs_max',
        'Perzentilspanne', 'Extremwertanteil', 'Max_Korrelation_Wert' # Include correlation if available
    ]

    # Use only existing features and handle NaN values for clustering
    current_features = [f for f in features_for_clustering if f in ergebnis_df.columns]
    data_for_clustering = ergebnis_df[current_features].fillna(0) # Fill NaN for clustering

    if data_for_clustering.empty or data_for_clustering.shape[0] < optimal_k:
        st.warning(f"Not enough data points ({data_for_clustering.shape[0]}) for {optimal_k} clusters. Setting n_clusters to number of data points.")
        optimal_k = data_for_clustering.shape[0] if data_for_clustering.shape[0] > 0 else 1

    if optimal_k == 0: # Handle case where optimal_k could become 0 if df is empty
        ergebnis_df['Cluster'] = -1
        return ergebnis_df

    # Normalize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_for_clustering)

    # K-Means Clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    ergebnis_df['Cluster'] = kmeans.fit_predict(scaled_data)

    return ergebnis_df


def apply_advanced_naming_and_recommendations(ergebnis_df_raw):
    """
    Names clusters and derives recommendations.
    Directly adapted from the user's script.
    """
    ergebnis_df = ergebnis_df_raw.copy()

    # Initialize columns if not already present
    if 'Clustername_final' not in ergebnis_df.columns:
        ergebnis_df['Clustername_final'] = 'unknown'
    if 'Empfehlung' not in ergebnis_df.columns:
        ergebnis_df['Empfehlung'] = 'unknown'
    if 'Korrelationspartner' not in ergebnis_df.columns:
        ergebnis_df['Korrelationspartner'] = np.nan
    if 'Korrelationswert' not in ergebnis_df.columns:
        ergebnis_df['Korrelationswert'] = np.nan


    # Calculate correlations before clustering, as this can influence a variable
    # Only for numeric columns in the original DF
    numeric_cols_original = st.session_state['original_df'].select_dtypes(include=np.number).columns
    if 'Time' in numeric_cols_original: # Exclude time if it became numeric somehow
        numeric_cols_original = numeric_cols_original.drop('Time')

    if not numeric_cols_original.empty and len(numeric_cols_original) > 1: # Ensure at least 2 numeric columns for correlation
        corr_matrix = st.session_state['original_df'][numeric_cols_original].corr().abs()
        np.fill_diagonal(corr_matrix.values, 0) # Set diagonal to 0 for self-correlation

        # Ensure 'Max_Korrelation_Wert' column exists for features
        if 'Max_Korrelation_Wert' not in ergebnis_df.columns:
            ergebnis_df['Max_Korrelation_Wert'] = np.nan

        for index, row in ergebnis_df.iterrows():
            var_name = row['Name']
            if var_name in corr_matrix.columns:
                # Find the strongest correlation partner
                max_corr_value = corr_matrix[var_name].max()
                if max_corr_value > 0.95: # Threshold for strong correlation
                    strongest_partner = corr_matrix[var_name].idxmax()
                    ergebnis_df.loc[index, 'Korrelationspartner'] = strongest_partner
                    ergebnis_df.loc[index, 'Korrelationswert'] = max_corr_value
                    ergebnis_df.loc[index, 'Max_Korrelation_Wert'] = max_corr_value # Update this for clustering input
                else:
                    ergebnis_df.loc[index, 'Korrelationspartner'] = np.nan
                    ergebnis_df.loc[index, 'Korrelationswert'] = np.nan
                    ergebnis_df.loc[index, 'Max_Korrelation_Wert'] = 0.0 # Default to 0 if no strong correlation
    else:
        st.info("No numeric columns found for correlation calculation or only one numeric column.")
        # Ensure these columns exist even if no correlation is calculated
        if 'Max_Korrelation_Wert' not in ergebnis_df.columns:
            ergebnis_df['Max_Korrelation_Wert'] = 0.0
        if 'Korrelationspartner' not in ergebnis_df.columns:
            ergebnis_df['Korrelationspartner'] = np.nan
        if 'Korrelationswert' not in ergebnis_df.columns:
            ergebnis_df['Korrelationswert'] = np.nan


    cluster_names = {}
    # Ensure 'Cluster' column exists before iterating over its unique values
    # This addresses the KeyError: 'Cluster' if clustering was skipped
    if 'Cluster' not in ergebnis_df.columns:
        ergebnis_df['Cluster'] = -1 # Assign a default cluster ID for unclustered items

    for cluster_id in sorted(ergebnis_df['Cluster'].unique()):
        if cluster_id == -1: # Handle unclustered/error case
            cluster_names[cluster_id] = "Not Assigned"
            continue

        cluster_data = ergebnis_df[ergebnis_df['Cluster'] == cluster_id]
        if cluster_data.empty:
            cluster_names[cluster_id] = f"Cluster {cluster_id}: empty"
            continue

        # Calculate average features in the cluster
        avg_features = cluster_data[['mean', 'std', 'min', 'max', 'unique_values', 'num_changes',
                                     'pct_changes', 'Perzentilspanne', 'Extremwertanteil']].mean()

        # Most frequent categories in the cluster
        top_category = cluster_data['Kategorie'].mode()[0] if not cluster_data['Kategorie'].empty else 'unknown'

        # Naming logic based on percentiles and statistical rules
        name_parts = []

        # Percentile Rule (Examples)
        if avg_features['Perzentilspanne'] < 0.01:
            name_parts.append("constant")
        if avg_features['Extremwertanteil'] > 0.1:
            name_parts.append("outlier_prone")
        if avg_features['pct_changes'] > 0.5:
            name_parts.append("dynamic")
        if avg_features['unique_values'] < 5 and avg_features['num_changes'] < 20:
            name_parts.append("status_like")

        # Statistical Rule (Examples)
        if avg_features['std'] == 0:
            name_parts.append("constant_value")
        elif avg_features['std'] < 0.1 and avg_features['unique_values'] <= 3:
            name_parts.append("binary/boolean")
        elif avg_features['std'] > 10 and avg_features['pct_changes'] > 0.7:
            name_parts.append("highly_variable")

        # Combine with most frequent category
        if top_category != 'unknown' and top_category not in name_parts:
            name_parts.append(top_category)

        # Fusion: If no specific rules apply, use most frequent category or "other"
        if not name_parts:
            cluster_name = f"Cluster {cluster_id}: {top_category if top_category != 'unknown' else 'other'}"
        else:
            cluster_name = f"Cluster {cluster_id}: {', '.join(sorted(set(name_parts)))}" # Remove duplicates and sort

        cluster_names[cluster_id] = cluster_name

    ergebnis_df['Clustername_final'] = ergebnis_df['Cluster'].map(cluster_names)

    # NEW Recommendation Logic (directly adapted from the Mermaid Diagram and text)
    # Reordered priorities to ensure 'review' takes precedence for problematic variables
    def get_recommendation(row):
        cluster_name = row['Clustername_final']
        # Ensure 'Cluster' column exists before accessing it
        cluster_id_val = row['Cluster'] if 'Cluster' in row else -1
        # Access ergebnis_df from the outer scope for cluster_size calculation
        # This assumes ergebnis_df is the DataFrame being processed by apply()
        cluster_size = len(ergebnis_df[ergebnis_df['Cluster'] == cluster_id_val])
        
        # Priority 1 (Highest): Keywords for review (e.g., unstable, outliers)
        if any(keyword in cluster_name.lower() for keyword in ["mode", "outlier", "unstable", "error_prone"]):
            return "review" # This should take precedence
        
        # Priority 2: Review based on cluster size
        if cluster_size == 1:
            return "uncertain / review individually"
        if cluster_size == 2:
            return "review (small cluster)"

        # Priority 3: Keywords for ignoring
        if any(keyword in cluster_name.lower() for keyword in ["static", "configuration_value", "constant", "non-numeric", "noise signal"]):
            return "better ignore"
            
        # Priority 4: Specific keywords for high relevance
        if any(keyword in cluster_name.lower() for keyword in ["energiesensor", "digitaletrigger", "dynamic", "variable", "counter"]):
            return "highly relevant"
        
        # Fallback
        return "further review"

    ergebnis_df['Empfehlung'] = ergebnis_df.apply(get_recommendation, axis=1)

    # Post-processing for highly correlated variables
    for index, row in ergebnis_df.iterrows():
        if pd.notna(row['Korrelationspartner']) and row['Korrelationswert'] > 0.95:
            # If it's already a 'review' type recommendation, don't downgrade it.
            # Only change to 'highly relevant (correlated)' if it's not already a 'review' or 'highly relevant'
            if row['Empfehlung'] not in ["highly relevant", "review", "uncertain / review individually", "review (small cluster)"]:
                ergebnis_df.loc[index, 'Empfehlung'] = "highly relevant (correlated)"

    return ergebnis_df


def create_correlation_heatmap(df):
    """
    Creates a correlation heatmap for numeric variables.
    Directly adapted from the user's script (Matplotlib/Seaborn).
    """
    numeric_df = df.select_dtypes(include=np.number)
    if 'Time' in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=['Time'])

    if numeric_df.empty:
        st.info("No numeric data found to create the correlation heatmap.")
        return

    corr_matrix = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap of Numeric Variables")
    st.pyplot(fig)


def create_batch_diagrams(df_original):
    """
    Creates batch diagrams for numeric variables (Fireplot).
    Directly adapted from the user's script (Matplotlib).
    """
    st.subheader("Batch Diagrams: Time Series of Numeric Variables")

    time_column = None
    for col in df_original.columns:
        if col.strip().lower() == 'time':
            time_column = col
            break

    if time_column is None:
        st.error("Time column (e.g., 'time') not found.")
        return

    try:
        df_original[time_column] = pd.to_datetime(df_original[time_column], errors='coerce')
        df_original = df_original.dropna(subset=[time_column])
        df_original = df_original.sort_values(by=time_column)
    except Exception as e:
        st.error(f"Error converting time column: {e}")
        return

    numeric_columns = df_original.select_dtypes(include='number').columns.tolist()
    numeric_columns = [col for col in numeric_columns if col != time_column]

    if not numeric_columns:
        st.info("No numeric columns found to create batch diagrams.")
        return

    # Optional: Only a sample (e.g., 100000 rows) to save resources
    sample_df = df_original[[time_column] + numeric_columns].copy()
    sample_df = sample_df.head(100000)

    cols_per_fig = 5 # Adjusted for Streamlit display
    rows_per_fig = 5 # Adjusted for Streamlit display
    plots_per_fig = cols_per_fig * rows_per_fig

    total_vars = len(numeric_columns)
    
    st.write(f"Generating {total_vars} diagrams. This may take a while.")

    # Use Streamlit columns for better layout of multiple plots if needed, or loop with st.pyplot
    for i in range(0, total_vars, plots_per_fig):
        sub_vars = numeric_columns[i:i+plots_per_fig]
        fig, axes = plt.subplots(rows_per_fig, cols_per_fig, figsize=(20, 10), sharex=True)
        axes = axes.flatten()

        for j, var in enumerate(sub_vars):
            ax = axes[j]
            ax.plot(sample_df[time_column], sample_df[var], linewidth=0.5)
            ax.set_title(var, fontsize=9)
            ax.tick_params(axis='x', labelrotation=45, labelsize=7)
            ax.tick_params(axis='y', labelsize=7)

        for k in range(len(sub_vars), len(axes)):
            axes[k].axis('off')

        plt.suptitle("Time Series of Numeric Variables", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        st.pyplot(fig)
        plt.close(fig) # Close figure to free memory


# --- Streamlit App Structure ---
st.title("Variable Analysis and Classification (Final Version)")
st.markdown("This application helps you classify variables in your dataset and provide recommendations based on their statistical characteristics and clustering.")
st.markdown("---")

# Session State Initialization
if 'original_df' not in st.session_state:
    st.session_state['original_df'] = None
if 'ergebnis_df' not in st.session_state:
    st.session_state['ergebnis_df'] = None

# --- Step 1: Upload Data ---
st.header("Step 1: Upload Data")
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xls", "xlsx", "xlsm"])

if uploaded_file is not None:
    st.info("File uploaded. Processing...")
    st.session_state['original_df'] = load_data_robustly(uploaded_file)
    if st.session_state['original_df'] is not None:
        st.success("✅ Data successfully loaded into session.")
    else:
        st.error("❌ Data could not be loaded. Please check the file.")
else:
    st.info("Please upload a file to begin.")

st.markdown("---")

# --- Step 2: Perform Analysis ---
st.header("Step 2: Perform Analysis and Classification")
if st.session_state['original_df'] is not None:
    if st.button("Start Analysis"):
        with st.spinner("Calculating features, classifying, and clustering..."):
            df_cleaned = st.session_state['original_df'].copy()

            # Identify numeric and non-numeric columns based on original_df
            numeric_cols = df_cleaned.select_dtypes(include=np.number).columns.tolist()
            non_numeric_cols = df_cleaned.select_dtypes(exclude=np.number).columns.tolist()

            all_features_list = []

            # 2.1: Feature Calculation for numeric columns
            if numeric_cols:
                numeric_ergebnis_df = calculate_features(df_cleaned[numeric_cols])
                all_features_list.append(numeric_ergebnis_df)
            else:
                st.warning("No numeric columns found for feature calculation.")

            # 2.2: Prepare features for non-numeric columns
            if non_numeric_cols:
                non_numeric_features_data = []
                # Define all possible feature columns to ensure consistency
                all_feature_columns = [
                    'Name', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'range',
                    'variance', 'median', 'mad', 'skewness', 'kurtosis', 'unique_values',
                    'num_changes', 'pct_changes', 'pct_zeros', 'pct_nans', 'abs_max',
                    'change_frequency', 'max_consecutive_same', 'min_consecutive_same',
                    'Perzentilspanne', 'Extremwertanteil'
                ]
                
                for col in non_numeric_cols:
                    row_data = {'Name': col, 'Kategorie': 'non-numeric'} 
                    row_data['count'] = df_cleaned[col].count()
                    row_data['unique_values'] = df_cleaned[col].nunique()
                    row_data['pct_nans'] = df_cleaned[col].isnull().sum() / len(df_cleaned[col]) if len(df_cleaned[col]) > 0 else np.nan
                    
                    # Fill all other feature columns with NaN for non-numeric variables
                    for feat_col in all_feature_columns:
                        if feat_col not in row_data:
                            row_data[feat_col] = np.nan
                    non_numeric_features_data.append(row_data)

                non_numeric_ergebnis_df = pd.DataFrame(non_numeric_features_data)
                all_features_list.append(non_numeric_ergebnis_df)
            else:
                st.info("No non-numeric columns found.")

            # Concatenate all feature dataframes
            if all_features_list:
                ergebnis_df = pd.concat(all_features_list, ignore_index=True)
            else:
                ergebnis_df = pd.DataFrame() # Empty DataFrame if no columns processed

            # Step 2.3: Categorize variables (applies to both numeric and non-numeric)
            if not ergebnis_df.empty:
                ergebnis_df = categorize_variables(ergebnis_df)
            else:
                st.warning("No data for categorization.")
                st.session_state['ergebnis_df'] = pd.DataFrame(columns=['Name', 'Kategorie', 'Cluster', 'Clustername_final', 'Empfehlung', 'Korrelationspartner', 'Korrelationswert'])
                st.success("✅ Analysis and classification completed, but no variables processed.")
                # No return here, allow the rest of the script to run with empty/default data

            # Initialize Cluster column before clustering logic to prevent KeyError
            # This block was moved here to ensure 'Cluster' and 'Clustername_final' are always present
            if 'Cluster' not in ergebnis_df.columns:
                ergebnis_df['Cluster'] = -1 # Default to -1 for unclustered
            if 'Clustername_final' not in ergebnis_df.columns:
                ergebnis_df['Clustername_final'] = 'Not Applicable' # Default name
            # Also initialize 'Max_Korrelation_Wert' here if it's not present for clustering
            if 'Max_Korrelation_Wert' not in ergebnis_df.columns:
                ergebnis_df['Max_Korrelation_Wert'] = 0.0


            # Step 2.4: Clustering (only if there are numeric features and enough data points)
            # Filter for variables that are candidates for clustering (i.e., not 'non-numeric' or 'non-numeric/empty')
            clustering_candidates = ergebnis_df[
                (ergebnis_df['Kategorie'] != 'non-numeric') &
                (ergebnis_df['Kategorie'] != 'non-numeric/empty') &
                (ergebnis_df['count'] > 0) # Only cluster non-empty numeric data
            ].copy()

            if not clustering_candidates.empty and clustering_candidates.shape[0] > 1:
                # Prepare data for clustering, ensure all relevant columns are numeric and not NaN
                features_for_clustering_check = [
                    'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'range',
                    'variance', 'median', 'mad', 'skewness', 'kurtosis', 'unique_values',
                    'num_changes', 'pct_changes', 'pct_zeros', 'pct_nans', 'abs_max',
                    'Perzentilspanne', 'Extremwertanteil', 'Max_Korrelation_Wert' # Include correlation if available
                ]
                # Ensure only actual numeric columns are selected for scaling, fill NaNs
                clustering_data_subset = clustering_candidates[[f for f in features_for_clustering_check if f in clustering_candidates.columns]].fillna(0)

                if not clustering_data_subset.empty and clustering_data_subset.shape[0] > 1:
                    optimal_k = find_optimal_clusters_kmeans(clustering_data_subset)
                    st.info(f"Optimal number of clusters found: {optimal_k}")
                    
                    ergebnis_df_clustered_part = apply_kmeans_clustering(clustering_candidates, optimal_k)
                    
                    # Update 'Cluster' column in main ergebnis_df based on 'Name'
                    ergebnis_df = ergebnis_df.set_index('Name')
                    ergebnis_df.update(ergebnis_df_clustered_part.set_index('Name')[['Cluster']]) # Only update 'Cluster' column
                    ergebnis_df = ergebnis_df.reset_index()
                else:
                    st.warning("Not enough valid numeric variables for clustering. Clustering skipped.")
            else:
                st.warning("Not enough numeric variables for clustering. Clustering skipped.")

            # Step 2.5: Advanced Naming and Recommendations
            st.session_state['ergebnis_df'] = apply_advanced_naming_and_recommendations(ergebnis_df)

        st.success("✅ Analysis and classification completed.")
        st.subheader("Classification Results Table:")
        st.dataframe(st.session_state['ergebnis_df'])
else:
    st.info("Please upload data in Step 1 first to perform analysis.")

st.markdown("---")

# --- Step 3: Visualize Results ---
st.header("Step 3: Visualize Results")
if st.session_state.get('ergebnis_df') is not None and not st.session_state['ergebnis_df'].empty:

    st.subheader("1. Variable Distribution by Category")
    try:
        category_counts = st.session_state['ergebnis_df']['Kategorie'].value_counts().reset_index()
        category_counts.columns = ['Category', 'Count'] 
        
        fig_category = px.bar(category_counts, x='Category', y='Count',
                              title='Distribution of Variables by Category',
                              labels={'Category': 'Category', 'Count': 'Number of Variables'},
                              color='Category',
                              color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_category, use_container_width=True)
    except Exception as e:
        st.error(f"Error generating category chart: {e}")

    st.markdown("---")

    st.subheader("2. Hierarchical Grouping of Variables (Sunburst Chart)")
    with st.spinner("Generating Sunburst chart..."):
        try:
            # Ensure necessary columns for Sunburst exist
            sunburst_df = st.session_state['ergebnis_df'].copy()
            # Rename columns for the English path in the Sunburst chart
            sunburst_df = sunburst_df.rename(columns={'Kategorie': 'Category', 'Clustername_final': 'Clustername', 'Empfehlung': 'Recommendation'})
            for col in ['Category', 'Clustername', 'Recommendation', 'Name']:
                if col not in sunburst_df.columns:
                    sunburst_df[col] = 'N/A' # Placeholder if column is missing

            fig_sunburst = px.sunburst(sunburst_df,
                                       path=['Category', 'Clustername', 'Recommendation', 'Name'],
                                       title='Hierarchical Grouping of Variables',
                                       color='Category', 
                                       color_discrete_sequence=px.colors.qualitative.Pastel 
                                      )
            fig_sunburst.update_layout(margin=dict(t=50, l=0, r=0, b=0), 
                                       hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"), 
                                       height=700 
                                      )
            st.plotly_chart(fig_sunburst, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating Sunburst chart: {e}")
    st.markdown("---")

    st.subheader("3. Distribution of Recommendations (Pie Chart)")
    with st.spinner("Generating recommendation pie chart..."):
        try:
            if 'Empfehlung' in st.session_state['ergebnis_df'].columns:
                empfehlung_counts = st.session_state['ergebnis_df']["Empfehlung"].value_counts().reset_index()
                empfehlung_counts.columns = ['Recommendation', 'Count'] 
                
                # Sort by 'Recommendation' to ensure consistent order
                empfehlung_counts = empfehlung_counts.sort_values(by='Recommendation')

                fig_pie = px.pie(empfehlung_counts, 
                                 values='Count', 
                                 names='Recommendation', 
                                 title='Distribution of Recommendations',
                                 color_discrete_sequence=px.colors.qualitative.Pastel 
                                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label',
                                      marker=dict(line=dict(color='#000000', width=1))) 
                fig_pie.update_layout(height=400,
                                      title_font_size=16, 
                                      legend_font_size=12, 
                                      uniformtext_minsize=10, uniformtext_mode='hide') 
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("Recommendation column not found for chart generation.")
        except Exception as e:
            st.error(f"Error generating recommendation pie chart: {e}")
    st.markdown("---")

    st.subheader("4. Batch Diagrams of Numeric Variables")
    with st.spinner("Generating batch diagrams..."):
        try:
            if st.session_state.get('original_df') is not None:
                create_batch_diagrams(st.session_state['original_df'].copy())
            else:
                st.info("Original data not available for batch diagram generation.")
        except Exception as e:
            st.error(f"Error generating batch diagrams: {e}")
    st.markdown("---")

    st.subheader("5. Correlation Heatmap of Numeric Variables")
    with st.spinner("Generating correlation heatmap..."):
        try:
            if st.session_state.get('original_df') is not None:
                create_correlation_heatmap(st.session_state['original_df'])
            else:
                st.info("Original data not available for correlation heatmap.")
        except Exception as e:
            st.error(f"Error generating correlation heatmap: {e}")
else:
    st.info("Please run Step 2 first to generate classified data for visualizations.")

st.markdown("---")

# --- Step 4: Download Final Classified Data ---
st.header("Step 4: Download Final Classified Data")
if st.session_state.get('ergebnis_df') is not None and not st.session_state['ergebnis_df'].empty:
    st.write("The processed data with classifications is ready for download.")
    csv_file = st.session_state['ergebnis_df'].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Classified Data (CSV)",
        data=csv_file,
        file_name="classified_data.csv",
        mime="text/csv",
    )
else:
    st.info("Please run Step 2 first to generate classified data.")

st.markdown("---")

# --- Step 5: Generate Data Report ---
st.header("Step 5: Generate Data Report")
if st.session_state.get('original_df') is not None and not st.session_state['original_df'].empty:
    st.write("Due to library incompatibility with Streamlit Cloud's Python version (Python 3.13), "
             "comprehensive data profiling libraries like `ydata-profiling` and `sweetviz` are currently not supported. "
             "Below is a basic in-app data overview.")
    
    st.subheader("Basic Data Overview")
    
    st.write("### Dataset Information (df.info())")
    # Redirect info() output to a string buffer to display in Streamlit
    buffer = io.StringIO()
    st.session_state['original_df'].info(buf=buffer)
    st.text(buffer.getvalue())

    st.write("### Descriptive Statistics (df.describe())")
    st.dataframe(st.session_state['original_df'].describe())

    st.write("### Missing Values")
    st.dataframe(st.session_state['original_df'].isnull().sum().rename('Missing Count'))

    st.write("### Unique Values for Non-Numeric Columns (Top 10)")
    non_numeric_cols_for_report = st.session_state['original_df'].select_dtypes(exclude=np.number).columns
    if not non_numeric_cols_for_report.empty:
        for col in non_numeric_cols_for_report:
            st.write(f"#### Column: `{col}`")
            st.dataframe(st.session_state['original_df'][col].value_counts().head(10))
    else:
        st.info("No non-numeric columns found for unique value analysis.")

else:
    st.info("Please upload data in Step 1 first to generate a data profile report.")

st.markdown("---")
st.info("Thank you for using the Variable Classification App!")
