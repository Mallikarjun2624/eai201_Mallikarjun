# fifa_week4_full_app.py
# Week-4 Final Project: Full FIFA 2026 Simulation & Prediction App

import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter

# ---------------------------------------
# Streamlit Config
# ---------------------------------------
st.set_page_config(page_title="FIFA Week 4 - Full Tournament Simulator", layout="wide")
st.title("🏆 FIFA World Cup 2026 — Full Project (Qualification to Winner)")
st.caption("Predicting the next FIFA World Cup Champion using ML-based simulation")

# ---------------------------------------
# Load and Prepare Data
# ---------------------------------------
@st.cache_data
def load_cleaned_data(path="cleaned_data.xlsx"):
    if os.path.exists(path):
        try:
            return pd.read_excel(path)
        except Exception as e:
            st.error(f"Error reading {path}: {e}")
            return None
    alt = "/mnt/data/cleaned_data.xlsx"
    if os.path.exists(alt):
        try:
            return pd.read_excel(alt)
        except Exception as e:
            st.error(f"Error reading {alt}: {e}")
            return None
    return None

def build_fallback_df():
    teams = [
        "Brazil","Argentina","France","England","Germany","Spain","Portugal","Netherlands",
        "USA","Mexico","Japan","South Korea","Australia","Morocco","Croatia","Uruguay",
        "Switzerland","Senegal","Iran","Qatar","Canada","Cameroon","Ghana","Serbia",
        "Poland","Denmark","Tunisia","Saudi Arabia","Ukraine","Chile","Egypt","Norway",
        "Nigeria","Turkey","Wales","Scotland","Ecuador","Paraguay","Costa Rica","Iceland",
        "Algeria","Czech Republic","Greece","Romania","Ireland","Slovakia","Finland","China"
    ]
    np.random.seed(42)
    df = pd.DataFrame({
        "Team": teams,
        "FIFA_Rank": list(range(1,49)),
        "Avg_Age": np.random.randint(24,31,48),
        "WinPct": np.random.randint(40,85,48),
        "GoalDiff": np.random.randint(-5,25,48),
        "Experience": np.random.randint(20,80,48),
        "Host": [1 if t in ("USA","Canada","Mexico") else 0 for t in teams],
        "Qualified_2026": np.random.choice([0,1],48)
    })
    return df

def merge_with_fallback(df_loaded, fallback_df):
    if df_loaded is None:
        return fallback_df
    if len(df_loaded) < 48:
        st.sidebar.warning(f"Loaded dataset has {len(df_loaded)} rows — filling missing teams.")
        df_merged = pd.concat([df_loaded, fallback_df])
        df_merged = df_merged.drop_duplicates(subset=["Team"], keep="first").reset_index(drop=True)
        return df_merged
    return df_loaded

def prepare_X_y(df, feature_cols=None, target_col="Qualified_2026"):
    if feature_cols is None:
        candidates = ["FIFA_Rank","Avg_Age","WinPct","GoalDiff","Experience","Host"]
        feature_cols = [c for c in candidates if c in df.columns]
    X = df[feature_cols].copy()
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(imputer.fit_transform(X))
    y = df[target_col].astype(int).values if target_col in df.columns else np.random.choice([0,1], len(df))
    return X_scaled, y, feature_cols, imputer, scaler

# ---------------------------------------
# Model Training & Evaluation
# ---------------------------------------
def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:,1] if hasattr(clf, "predict_proba") else np.zeros_like(y_pred)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba) if len(np.unique(y_test))>1 else 0.0,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred, zero_division=0)
    }

def train_and_compare_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42)
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = {"model": model, "metrics": evaluate_model(model, X_test, y_test)}
    return results

def pick_best_model(results):
    best_model, best_name, best_score = None, None, -1
    for name, info in results.items():
        score = info["metrics"]["roc_auc"]
        if score > best_score:
            best_model, best_name, best_score = info["model"], name, score
    return best_name, best_model, best_score

# ---------------------------------------
# Tournament Simulation (Enhanced 48 Teams)
# ---------------------------------------
def logits_from_probas(probs):
    eps = 1e-6
    return np.log((probs + eps) / (1 - probs + eps))

def simulate_tournament(teams, logits_map, deterministic=False):
    """
    Full 48-team knockout tournament:
    - Round of 48
    - Round of 32
    - Round of 16
    - Quarterfinals
    - Semifinals
    - Final
    Returns all stage results and champion.
    """
    np.random.shuffle(teams)

    def knockout_round(participants, stage_name):
        winners, matches = [], []
        n = len(participants)
        np.random.shuffle(participants)
        for i in range(0, n - 1, 2):
            a, b = participants[i], participants[i + 1]
            pa = 1 / (1 + np.exp(-(logits_map[a] - logits_map[b])))
            winner = a if (deterministic and pa >= 0.5) or (not deterministic and np.random.rand() < pa) else b
            winners.append(winner)
            matches.append((a, b, winner, pa))
        if n % 2 == 1:  # handle odd number
            bye = participants[-1]
            winners.append(bye)
            matches.append((bye, "BYE", bye, 1.0))
        return winners, matches

    stages = {}
    current_teams = teams
    round_names = ["Round of 48", "Round of 32", "Round of 16", "Quarterfinals", "Semifinals", "Final"]

    for stage in round_names:
        winners, matches = knockout_round(current_teams, stage)
        stages[stage] = matches
        current_teams = winners
        if len(winners) == 1:
            break

    champion = current_teams[0]
    return stages, champion

# ---------------------------------------
# Load Dataset
# ---------------------------------------
st.sidebar.header("Controls")
df_loaded = load_cleaned_data()
df_final = merge_with_fallback(df_loaded, build_fallback_df())
st.sidebar.success(f"✅ Dataset loaded with {len(df_final)} teams")

team_column = "Team" if "Team" in df_final.columns else df_final.columns[0]

# ---------------------------------------
# Tabs
# ---------------------------------------
tab_data, tab_model, tab_sim, tab_feat = st.tabs([
    "📁 Data Overview", "⚙ Model & Evaluation", "🏆 Tournament Simulation", "📊 Feature Importance"
])

# ------------------ Data Overview ------------------
with tab_data:
    st.header("Data Overview")
    st.dataframe(df_final, use_container_width=True, height=500)
    st.write(f"*Total Teams:* {len(df_final)}")
    st.write("*Columns:*", list(df_final.columns))
    st.write(df_final.describe(include="all").T)

# ------------------ Model & Evaluation ------------------
# ------------------ Model & Evaluation ------------------
with tab_model:
    st.header("Model Training & Evaluation")
    
    # Select features automatically from dataset
    features = [c for c in ["FIFA_Rank", "Avg_Age", "WinPct", "GoalDiff", "Experience", "Host"] if c in df_final.columns]
    X, y, feature_cols, imp, sca = prepare_X_y(df_final, features)
    
    test_size = st.slider("Test set fraction", 0.1, 0.4, 0.25)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    if st.button("Train & Evaluate Models"):
        # --- Define only the selected models ---
        models = {
            "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42)
        }

        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else np.zeros_like(y_pred)

            metrics = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1": f1_score(y_test, y_pred, zero_division=0),
                "ROC_AUC": roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.0,
            }
            results[name] = {"model": model, "metrics": metrics}

        # --- Display metrics ---
        st.subheader("Model Evaluation Results")
        metrics_df = pd.DataFrame(
            {name: m["metrics"] for name, m in results.items()}
        ).T.sort_values(by="ROC_AUC", ascending=False)
        st.dataframe(metrics_df.style.format("{:.3f}"))

        # --- Select the best model ---
        best_name = metrics_df.index[0]
        best_model = results[best_name]["model"]
        best_score = metrics_df.loc[best_name, "ROC_AUC"]

        st.success(f"🏅 Best Model: {best_name} (ROC_AUC = {best_score:.3f})")
        st.session_state["best_model"] = best_model
        st.session_state["imputer"] = imp
        st.session_state["scaler"] = sca
        st.session_state["feature_cols"] = feature_cols

        # --- Plot ROC-AUC comparison ---
        st.markdown("### 📊 Model ROC-AUC Comparison")
        fig, ax = plt.subplots()
        ax.bar(metrics_df.index, metrics_df["ROC_AUC"], color=["#2980B9", "#27AE60"])
        ax.set_ylabel("ROC-AUC Score")
        ax.set_title("Model Performance Comparison (Without Gradient Boosting)")
        st.pyplot(fig)

# ------------------ Tournament Simulation ------------------
with tab_sim:
    st.header("🏆 Tournament Simulation")
    if "best_model" not in st.session_state:
        st.warning("Please train models first.")
    else:
        bm = st.session_state["best_model"]
        imp = st.session_state["imputer"]
        sca = st.session_state["scaler"]
        fcols = st.session_state["feature_cols"]

        X_all = df_final[fcols]
        X_all_scaled = sca.transform(imp.transform(X_all))
        probs = bm.predict_proba(X_all_scaled)[:,1]
        df_prob = df_final.copy()
        df_prob["Model_Prob"] = probs
        df_prob = df_prob.sort_values("Model_Prob", ascending=False).reset_index(drop=True)

        st.subheader("Team Ranking by Model Probability (Top 20)")
        st.dataframe(df_prob[[team_column,"FIFA_Rank","Model_Prob"]].head(20))

        deterministic = st.checkbox("Deterministic (No randomness)", value=False)
        if st.button("Run Simulation"):
            logits = logits_from_probas(df_prob["Model_Prob"].values)
            logits_map = {df_prob[team_column].iloc[i]: logits[i] for i in range(len(df_prob))}
            teams = df_prob[team_column].tolist()

            stages, champion = simulate_tournament(teams, logits_map, deterministic)
            st.success(f"🏆 Champion: *{champion}*")

            for stage, matches in stages.items():
                st.markdown(f"### {stage}")
                st.dataframe(pd.DataFrame(matches, columns=["Team A", "Team B", "Winner", "P(A Wins)"]))

            if "Semifinals" in stages:
                sf = pd.DataFrame(stages["Semifinals"], columns=["Team A","Team B","Winner","P(A Wins)"])
                finalists = sf["Winner"].tolist()
                st.info(f"🏅 Finalists: {', '.join(finalists)}")

            if "Final" in stages:
                final = stages["Final"][0]
                st.success(f"🏆 Final: {final[0]} vs {final[1]} → Winner: *{final[2]}* (P={final[3]:.2f})")

# ------------------ Feature Importance ------------------
with tab_feat:
    st.header("Feature Importance")
    if "best_model" not in st.session_state:
        st.warning("Train models first.")
    else:
        bm = st.session_state["best_model"]
        fcols = st.session_state["feature_cols"]
        if hasattr(bm, "feature_importances_"):
            fi = pd.DataFrame({"Feature": fcols, "Importance": bm.feature_importances_}).sort_values("Importance", ascending=False)
            st.bar_chart(fi.set_index("Feature"))
        elif hasattr(bm, "coef_"):
            fi = pd.DataFrame({"Feature": fcols, "Coef": bm.coef_[0]}).sort_values("Coef", ascending=False)
            st.bar_chart(fi.set_index("Feature"))
        else:
            st.info("This model does not provide feature importances.")

st.markdown("---")
st.caption("Developed by Mallikarjun H Biradar — Chanakya University | Week 4 Final Implementation")


#cd "C:\Users\Suresha\Downloads"
#streamlit run fifa_week4_full_app.py