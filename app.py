import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import plotly.express as px

st.set_page_config(page_title="Spy-Themed Financial ML", layout="wide")

# --- Sidebar: Data Inputs ---
st.sidebar.header("Data Sources")
kragle_file = st.sidebar.file_uploader("Upload Kragle CSV", type=["csv"])
ticker      = st.sidebar.text_input("Yahoo Finance ticker", value="AAPL")
start_date  = st.sidebar.date_input("Start date", value=pd.to_datetime("2020-01-01"))
end_date    = st.sidebar.date_input("End date",   value=pd.to_datetime("2025-05-15"))

df_kragle = pd.read_csv(kragle_file) if kragle_file else pd.DataFrame()
yf_data   = yf.download(ticker, start=start_date, end=end_date)

# --- Theme selector ---
theme = st.sidebar.selectbox("Choose your covert theme",
    ["James Bond","Agent 47","Mission Impossible","Jason Bourne","Jack Reacher"]
)

# --- Helper: CSS + GIF loader ---
def apply_theme(css, gif, title):
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    st.image(f"assets/{gif}", use_column_width=True)
    st.title(title)

# --- 1) James Bond → Linear Regression ---
if theme == "James Bond":
    css = """
      .stApp {background-color:#f0f0f0; color:#000; font-family:'sleek';}
      .css-1v3fvcr {background-image:url('');}
    """
    apply_theme(css, "jamesbond.gif", "🕶️ 007 Price Prediction")
    df = yf_data[["Close"]].dropna().reset_index()
    df["Day"] = np.arange(len(df))
    if df.empty or len(df) < 2:
        st.error("Not enough data returned from Yahoo Finance to train the model.")
    else:
        X, y = df[["Day"]], df["Close"]
        model = LinearRegression().fit(X, y)
        days_ahead = st.slider("Days ahead",1,30,7)
        future_X = np.arange(len(df),len(df)+days_ahead).reshape(-1,1)
        preds = model.predict(future_X)
        forecast = pd.DataFrame({
          "Date": pd.date_range(end=end_date, periods=days_ahead+1, closed="right"),
          "Price": preds
        })
        fig = px.line(df, x="Date",y="Close", title="Historical vs Forecast 🎯")
        fig.add_scatter(x=forecast["Date"], y=forecast["Price"], mode="lines", name="Predicted")
        st.plotly_chart(fig, use_container_width=True)

# --- 2) Agent 47 → Logistic Regression ---
elif theme == "Agent 47":
    css = """
      .stApp {background-color:#222; color:#fff; font-family:'stealth';}
    """
    apply_theme(css, "agent47.gif", "🔫 Stealthy Up/Down Classifier")
    df = yf_data.copy().dropna()
    df["FutureRet"] = df["Close"].pct_change().shift(-1)
    df["Up"] = (df["FutureRet"]>0).astype(int)
    feats = df[["Open","High","Low","Volume"]].iloc[:-1]
    target= df["Up"].iloc[:-1]
    clf = LogisticRegression(max_iter=200).fit(feats,target)
    st.write(f"**Accuracy:** {clf.score(feats,target):.2%}")
    st.bar_chart(pd.Series(clf.coef_[0], index=feats.columns))

# --- 3) Mission Impossible → K-Means Clustering ---
elif theme == "Mission Impossible":
    css = """
      .stApp {background-color:#001f3f; color:#39ff14; font-family:'digital';}
    """
    apply_theme(css, "mission_impossible.gif", "🛰️ Impossible Clusters")
    data = yf_data["Close"].pct_change().dropna().reset_index()
    X = data["Close"].values.reshape(-1,1)
    k = st.slider("Clusters",2,6,3)
    km = KMeans(n_clusters=k, random_state=42).fit(X)
    data["Cluster"] = km.labels_
    fig = px.scatter(data, x="Date", y="Close", color="Cluster",
                     title="Classified close-price cells")
    st.plotly_chart(fig, use_container_width=True)

# --- 4) Jason Bourne → Random Forest ---
elif theme == "Jason Bourne":
    css = """
      .stApp {background-color:#111; color:#ff4136; font-family:'gritty';}
    """
    apply_theme(css, "jasonbourne.gif", "🏃‍♂️ Bourne Feature Hunt")
    df = yf_data.copy().dropna()
    df["Ret"] = df["Close"].pct_change().shift(-1)
    df["Up"] = (df["Ret"]>0).astype(int)
    feats = df[["Open","High","Low","Volume"]].iloc[:-1]
    target= df["Up"].iloc[:-1]
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(feats,target)
    st.write(f"**Accuracy:** {rf.score(feats,target):.2%}")
    imp = pd.Series(rf.feature_importances_,index=feats.columns)
    st.bar_chart(imp)

# --- 5) Jack Reacher → SVM Classification ---
else:
    css = """
      .stApp {background-color:#fafafa; color:#2f4f4f; font-family:'rugged';}
    """
    apply_theme(css, "jackreacher.gif", "🔍 Reacher’s Support Vector")
    df = yf_data.copy().dropna()
    df["Ret"] = df["Close"].pct_change().shift(-1)
    df["Up"] = (df["Ret"]>0).astype(int)
    X, y = df[["Open","High","Low","Volume"]].iloc[:-1], df["Up"].iloc[:-1]
    svc = SVC(kernel='rbf', C=1.0).fit(X, y)
    st.write(f"**Accuracy:** {svc.score(X,y):.2%}")
    # no built-in importances; show a confusion matrix
    from sklearn.metrics import confusion_matrix
    preds = svc.predict(X)
    cm = confusion_matrix(y, preds)
    st.write("Confusion Matrix:")
    st.table(pd.DataFrame(cm, index=["Actual 0","Actual 1"], columns=["Pred 0","Pred 1"]))
s
