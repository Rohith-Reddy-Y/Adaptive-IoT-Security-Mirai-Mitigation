import os
import glob
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

# =========================
# CONFIG
# =========================
MODELS_ROOT = "IOT23_Models"   # keep same name in GitHub repo
THRESHOLD = 0.5
COMBINE_RULE = "mean"  # "max" or "mean"


# =========================
# Helpers
# =========================
def resolve_models_root(models_root: str) -> str:
    """
    Make MODELS_ROOT work both:
      - locally (running from project folder)
      - on Streamlit Cloud (/mount/src/<repo>/)
    """
    # 1) If it's already a valid absolute/relative path from current working dir
    p1 = os.path.abspath(models_root)
    if os.path.exists(p1):
        return p1

    # 2) Try relative to this file's directory (more reliable on Streamlit Cloud)
    here = os.path.dirname(os.path.abspath(__file__))
    p2 = os.path.join(here, models_root)
    if os.path.exists(p2):
        return os.path.abspath(p2)

    # 3) As a last fallback, return the "best guess" (for debug printing)
    return p1


def load_models(models_root: str):
    """
    Loads all models that have BOTH:
      - model.xgb
      - feature_list.txt
    Returns:
      (items, debug_lines)
    """
    items = []
    debug_lines = []

    abs_root = resolve_models_root(models_root)
    debug_lines.append(f"MODELS_ROOT (resolved): {abs_root}")

    if not os.path.exists(abs_root):
        debug_lines.append("MODELS_ROOT path does NOT exist.")
        return [], debug_lines

    ds_dirs = sorted(glob.glob(os.path.join(abs_root, "dataset*")))
    debug_lines.append(f"Found dataset folders: {len(ds_dirs)}")

    for ds_dir in ds_dirs:
        model_path = os.path.join(ds_dir, "model.xgb")
        feat_path = os.path.join(ds_dir, "feature_list.txt")

        if not os.path.exists(model_path):
            debug_lines.append(f"SKIP {os.path.basename(ds_dir)}: missing model.xgb")
            continue
        if not os.path.exists(feat_path):
            debug_lines.append(f"SKIP {os.path.basename(ds_dir)}: missing feature_list.txt")
            continue

        # load feature list
        with open(feat_path, "r", encoding="utf-8") as f:
            feats = [ln.strip() for ln in f if ln.strip()]

        # load model
        bst = xgb.Booster()
        bst.load_model(model_path)

        items.append({"name": os.path.basename(ds_dir), "bst": bst, "features": feats})
        debug_lines.append(f"LOADED {os.path.basename(ds_dir)}: features={len(feats)}")

    return items, debug_lines


def one_hot_proto_service(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "proto" in out.columns:
        p = out["proto"].astype(str).str.lower()
        out["proto_tcp"] = (p == "tcp").astype(float)
        out["proto_udp"] = (p == "udp").astype(float)
        out["proto_icmp"] = (p == "icmp").astype(float)

    if "service" in out.columns:
        s = out["service"].astype(str).str.lower()
        out["service_dns"] = (s == "dns").astype(float)
        out["service_http"] = (s == "http").astype(float)
        out["service_-"] = (s == "-").astype(float)

    return out


def prepare_features(df_raw: pd.DataFrame, feature_list: list[str]) -> pd.DataFrame:
    df = df_raw.copy()
    df = df.drop(columns=["y_binary", "y_multiclass"], errors="ignore")
    df = one_hot_proto_service(df)

    for c in feature_list:
        if c not in df.columns:
            df[c] = 0.0

    X = df[feature_list].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    return X


def predict_ensemble(df_raw: pd.DataFrame, models: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(models) == 0:
        raise RuntimeError(
            "No models found in MODELS_ROOT. Ensure IOT23_Models is in the GitHub repo and contains "
            "dataset*/model.xgb + dataset*/feature_list.txt."
        )

    per_probs = []
    for m in models:
        X = prepare_features(df_raw, m["features"])
        dmat = xgb.DMatrix(X, feature_names=m["features"])
        prob = m["bst"].predict(dmat)
        per_probs.append(prob)

    per_model_probs = np.vstack(per_probs).T  # (N, M)

    if COMBINE_RULE == "mean":
        final_prob = per_model_probs.mean(axis=1)
    else:
        final_prob = per_model_probs.max(axis=1)

    final_pred = (final_prob >= THRESHOLD).astype(int)
    return final_prob, final_pred, per_model_probs


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="IoT Attack Detector", layout="wide")
st.title("IoT Attack Detector (Ensemble of all trained models)")
st.caption("Upload RAW CSV/Parquet → predicts BENIGN/MALICIOUS using all dataset models (max/mean rule).")

# ---- EXTRA (debug only): show what Streamlit Cloud actually has
st.sidebar.subheader("Server file check")
st.sidebar.write("cwd:", os.getcwd())
try:
    st.sidebar.write("repo root:", os.listdir("."))
except Exception as e:
    st.sidebar.write("Could not list root:", str(e))
st.sidebar.write("IOT23_Models exists?", os.path.exists("IOT23_Models"))
if os.path.exists("IOT23_Models"):
    try:
        st.sidebar.write("IOT23_Models contents:", os.listdir("IOT23_Models")[:30])
    except Exception as e:
        st.sidebar.write("Could not list IOT23_Models:", str(e))


@st.cache_resource
def cached_models():
    models, debug = load_models(MODELS_ROOT)
    return models, debug


models, debug_lines = cached_models()

# Sidebar info
st.sidebar.header("Model Loader")
st.sidebar.write(f"Loaded models: **{len(models)}**")
st.sidebar.write(f"Combine rule: **{COMBINE_RULE}**")
st.sidebar.write(f"Threshold: **{THRESHOLD}**")

with st.sidebar.expander("Debug (model loading logs)"):
    st.code("\n".join(debug_lines) if debug_lines else "No debug logs")

if len(models) == 0:
    st.error("No models loaded. Check the sidebar Debug logs.")
    st.stop()

uploaded = st.file_uploader("Upload RAW data (CSV or Parquet)", type=["csv", "parquet"])

if uploaded is not None:
    if uploaded.name.lower().endswith(".csv"):
        df_raw = pd.read_csv(uploaded)
    else:
        df_raw = pd.read_parquet(uploaded)

    st.subheader("Preview (raw input)")
    st.dataframe(df_raw.head(30), use_container_width=True)
    st.write(f"Rows uploaded: **{len(df_raw)}**")

    if st.button("Run Detection"):
        final_prob, final_pred, per_model_probs = predict_ensemble(df_raw, models)

        out = df_raw.copy()
        out["final_prob"] = final_prob
        out["final_label"] = np.where(final_pred == 1, "MALICIOUS", "BENIGN")

        st.subheader("Results (top rows)")
        st.dataframe(out[["final_label", "final_prob"]].head(50), use_container_width=True)

        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download predictions.csv",
            data=csv_bytes,
            file_name="predictions.csv",
            mime="text/csv",
        )

        model_names = [m["name"] for m in models]
        best_idx = per_model_probs.argmax(axis=1)
        out2 = pd.DataFrame(
            {
                "best_model": [model_names[i] for i in best_idx],
                "best_model_prob": per_model_probs.max(axis=1),
            }
        )

        st.subheader("Most confident dataset-model (per row)")
        st.dataframe(out2.head(50), use_container_width=True)

        st.markdown("---")
        st.header("📊 Visual Analytics")

        benign_count = int((final_pred == 0).sum())
        mal_count = int((final_pred == 1).sum())
        total = len(final_pred)
        mal_rate = (mal_count / total) if total else 0.0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Rows", total)
        c2.metric("Benign", benign_count)
        c3.metric("Malicious", mal_count)
        c4.metric("Malicious Rate", f"{mal_rate*100:.2f}%")

        st.subheader("Probability Distribution (final_prob)")
        st.bar_chart(pd.Series(final_prob).value_counts(bins=30).sort_index())

        st.subheader("Benign vs Malicious (Counts)")
        st.bar_chart(pd.DataFrame({"count": [benign_count, mal_count]}, index=["BENIGN", "MALICIOUS"]))

        st.subheader("Top Models that Fired Most Often")
        top_models = out2["best_model"].value_counts().head(10)
        st.bar_chart(top_models)

        st.subheader("Most Confident Rows (Top 20)")
        out_conf = out.copy()
        out_conf["best_model"] = out2["best_model"]
        out_conf["best_model_prob"] = out2["best_model_prob"]
        out_conf = out_conf.sort_values("final_prob", ascending=False).head(20)
        st.dataframe(out_conf[["final_label", "final_prob", "best_model", "best_model_prob"]], use_container_width=True)

        st.subheader("Per-Model Probability Heatmap (first 30 rows)")
        show_n = min(30, per_model_probs.shape[0])
        heat_df = pd.DataFrame(per_model_probs[:show_n], columns=model_names)
        st.dataframe(heat_df.style.background_gradient(axis=None), use_container_width=True)
