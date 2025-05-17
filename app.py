import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import plotly.express as px

st.set_page_config(page_title="Spy-Themed Financial ML", layout="wide")

# --- Sidebar: Upload CSV ---
st.sidebar.header("Upload Financial Data CSV")
data_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

theme = st.sidebar.selectbox("Choose your covert theme",
    ["James Bond", "Agent 47", "Mission Impossible", "Jason Bourne", "Jack Reacher"]
)

# --- Apply Theme Function ---
def apply_theme(css, gif, title):
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    st.image(f"assets/{gif}", use_column_width=True)
    st.title(title)

# --- Convert Volume Strings like '52.16M' to floats ---
def convert_volume(val):
    if isinstance(val, str):
        val = val.strip().upper()
        try:
            if val.endswith("M"):
                return float(val[:-1]) * 1_000_000
            elif val.endswith("B"):
                return float(val[:-1]) * 1_000_000_000
            elif val.endswith("K"):
                return float(val[:-1]) * 1_000
            else:
                return float(val.replace(",", ""))
        except:
            return np.nan
    return val

# --- Main Logic ---
if data_file:
    df = pd.read_csv(data_file)

    # --- Clean and standardize column names ---
    df.columns = df.columns.str.strip().str.replace('"', '')
    col_map = {
        "Price": "Close",
        "Vol.": "Volume"
    }
    df = df.rename(columns=col_map)

    # --- Parse date ---
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
        df = df.dropna(subset=["Date"])  # Drop rows with invalid dates

    # --- Convert Volume to numeric ---
    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].apply(convert_volume)

    # --- Validate required columns ---
    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(df.columns):
        st.error(f"Missing required columns. Required: {', '.join(required)}")
        st.stop()

    df = df.sort_values("Date").dropna()

    # === Theme 1: James Bond ===
    if theme == "James Bond":
        css = ".stApp {background-color:#6b3b3bba; color:#6b3b3bba; font-family:'sleek';}"
        apply_theme(css, "jamesbond.gif", "🕶️ 007 Price Prediction")
        df_lr = df[["Date", "Close"]].dropna().reset_index(drop=True)
        df_lr["Day"] = np.arange(len(df_lr))

        if len(df_lr) < 2:
            st.error("Not enough data to train the model.")
        else:
            X, y = df_lr[["Day"]], df_lr["Close"]
            model = LinearRegression().fit(X, y)
            days_ahead = st.slider("Days ahead", 1, 30, 7)
            future_X = np.arange(len(df_lr), len(df_lr)+days_ahead).reshape(-1,1)
            preds = model.predict(future_X)
            forecast = pd.DataFrame({
                "Date": pd.date_range(start=df_lr["Date"].iloc[-1] + pd.Timedelta(days=1), periods=days_ahead),
                "Price": preds
            })
            fig = px.line(df_lr, x="Date", y="Close", title="Historical vs Forecast 🎯")
            fig.add_scatter(x=forecast["Date"], y=forecast["Price"], mode="lines", name="Predicted")
            st.plotly_chart(fig, use_container_width=True)

    # === Theme 2: Agent 47 ===
    elif theme == "Agent 47":
        css = ".stApp {background-color:#222; color:#fff; font-family:'stealth';}"
        apply_theme(css, "agent47.gif", "🔫 Stealthy Up/Down Classifier")
        df["FutureRet"] = df["Close"].pct_change().shift(-1)
        df["Up"] = (df["FutureRet"] > 0).astype(int)
        feats = df[["Open", "High", "Low", "Volume"]].iloc[:-1]
        target = df["Up"].iloc[:-1]
        clf = LogisticRegression(max_iter=200).fit(feats.values, target.values)
        st.write(f"**Accuracy:** {clf.score(feats.values, target.values):.2%}")
        st.bar_chart(pd.Series(clf.coef_[0], index=feats.columns))

    # === Theme 3: Mission Impossible ===
    elif theme == "Mission Impossible":
        css = ".stApp {background-color:#001f3f; color:#39ff14; font-family:'digital';}"
        apply_theme(css, "mission_impossible.gif", "🛰️ Impossible Clusters")
        df_cluster = df[["Date", "Close"]].copy()
        df_cluster["Return"] = df_cluster["Close"].pct_change()
        df_cluster = df_cluster.dropna().reset_index(drop=True)
        X = df_cluster["Return"].values.reshape(-1, 1)
        k = st.slider("Clusters", 2, 6, 3)
        km = KMeans(n_clusters=k, random_state=42).fit(X)
        df_cluster["Cluster"] = km.labels_
        fig = px.scatter(df_cluster, x="Date", y="Return", color="Cluster", title="Classified close-price cells")
        st.plotly_chart(fig, use_container_width=True)

    # === Theme 4: Jason Bourne ===
    elif theme == "Jason Bourne":
        css = ".stApp {background-color:#111; color:#ff4136; font-family:'gritty';}"
        apply_theme(css, "jasonbourne.gif", "🏃‍♂️ Bourne Feature Hunt")
        df["Ret"] = df["Close"].pct_change().shift(-1)
        df["Up"] = (df["Ret"] > 0).astype(int)
        feats = df[["Open", "High", "Low", "Volume"]].iloc[:-1]
        target = df["Up"].iloc[:-1]
        rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(feats.values, target.values)
        st.write(f"**Accuracy:** {rf.score(feats.values, target.values):.2%}")
        imp = pd.Series(rf.feature_importances_, index=feats.columns)
        st.bar_chart(imp)

    # === Theme 5: Jack Reacher ===
    else:
        css = ".stApp {background-color:#3b486bba; color:#1c23ff; font-family:'rugged';}"
        apply_theme(css, "jackreacher.gif", "🔍 Reacher’s Support Vector")
        df["Ret"] = df["Close"].pct_change().shift(-1)
        df["Up"] = (df["Ret"] > 0).astype(int)
        X = df[["Open", "High", "Low", "Volume"]].iloc[:-1]
        y = df["Up"].iloc[:-1]
        svc = SVC(kernel='rbf', C=1.0).fit(X.values, y.values)
        st.write(f"**Accuracy:** {svc.score(X.values, y.values):.2%}")
        preds = svc.predict(X.values)
        cm = confusion_matrix(y, preds)
        st.write("Confusion Matrix:")
        st.table(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))

else:
    st.warning("Please upload a CSV file to proceed.")
