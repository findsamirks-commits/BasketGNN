"""Streamlit Interactive Application for BasketGNN.

Next-Basket Recommendation & Basket-Completion Engine for Quick-Commerce Dark Stores.
Demonstrates LightGCN co-purchase topology, real-time Shelf-Vision OOS filtering,
and Fresh-Grade FEFO perishable clearance boosting.
"""

import json
import os
import sys
import pandas as pd
import streamlit as st

# Ensure project root is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.crosssell_bridge import CrossSellBridge
from src.data import RETAIL_CATALOG, load_dataset
from src.inference import BasketGNNPredictor

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="BasketGNN 🛒🧠 | Quick-Commerce RecSys",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .rec-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
        border-left: 4px solid #3B82F6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">🛒🧠 BasketGNN: Quick-Commerce Graph Recommender</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">'
    "LightGCN graph neural network on user–item co-purchase topology. "
    "Predicts <b>Next-Basket SKUs</b> and <b>Cart-Complementary Attachments</b> with real-time "
    "<b>Shelf-Vision OOS filtering</b> and <b>Fresh-Grade FEFO clearance boosting</b>."
    "</div>",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Cached Model Initialization
# -----------------------------------------------------------------------------
@st.cache_resource
def get_predictor():
    weights_path = os.path.join(ROOT_DIR, "runs", "basketgnn", "best.pt")
    return BasketGNNPredictor(
        weights_path=weights_path if os.path.exists(weights_path) else None,
        data_dir=os.path.join(ROOT_DIR, "data", "processed"),
    )

predictor = get_predictor()

# -----------------------------------------------------------------------------
# Sidebar: Operational Controls & Signals
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ RecSys Query Controls")

    user_id = st.number_input(
        "Select User ID",
        min_value=0,
        max_value=predictor.num_users - 1,
        value=42,
        step=1,
        help="Evaluates user's historical interaction subgraph",
    )

    query_mode = st.radio(
        "Prediction Mode",
        ["Basket Completion (Cart Cross-Sell)", "Next-Basket (User Only)"],
        index=0,
    )

    top_k = st.slider("Top-K Recommendations", min_value=3, max_value=15, value=5)

    st.markdown("---")
    st.subheader("🌐 Closed-Loop Signals")

    enable_oos = st.toggle(
        "🚫 Shelf-Vision OOS Filter",
        value=True,
        help="Zeroes out recommendations for items detected Out-of-Stock on shelf",
    )

    oos_skus = set()
    if enable_oos:
        # Default OOS: Amul Butter (ID 2) and Surf Excel (ID 26)
        all_item_names = [item["name"] for item in RETAIL_CATALOG]
        oos_selected = st.multiselect(
            "Simulate Out-of-Stock SKUs",
            all_item_names,
            default=["Amul Pasteurised Butter 500g"],
        )
        name_to_id = {item["name"]: item["id"] for item in RETAIL_CATALOG}
        oos_skus = {name_to_id[n] for n in oos_selected if n in name_to_id}

    enable_fefo = st.toggle(
        "🍅 Fresh-Grade FEFO Clearance Boost",
        value=True,
        help="Boosts priority score for near-peak produce flagged SELL_FIRST (2-4d remaining)",
    )

    fefo_skus = {}
    if enable_fefo:
        # Default FEFO items: Fresh Tomato (2.5d left) & Fresh Banana (3.0d left)
        fefo_skus = {10: 2.5, 11: 3.0}
        st.caption("Active FEFO boosts: Tomato (2.5d left), Banana (3.0d left)")

    st.markdown("---")
    st.markdown("### 👨‍💻 Developer & Architect")
    st.markdown("**Samir Sandhwar** | Category Head & AI Engineer")
    st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-findsamirks--commits-black?style=flat&logo=github)](https://github.com/findsamirks-commits/BasketGNN)")
    st.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-samirsandhwar-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/samirsandhwar)")

# -----------------------------------------------------------------------------
# Main Panel: Cart Builder & History
# -----------------------------------------------------------------------------
col_left, col_right = st.columns([0.45, 0.55])

with col_left:
    st.subheader("🛒 Active Shopping Cart Context")

    if query_mode == "Basket Completion (Cart Cross-Sell)":
        catalog_names = [item["name"] for item in RETAIL_CATALOG]
        default_cart = [
            "Amul Taaza Fresh Toned Milk 1L",
            "Britannia Daily Fresh White Bread 400g",
        ]

        active_cart_names = st.multiselect(
            "Items currently added to cart:",
            catalog_names,
            default=default_cart,
        )

        name_to_id = {item["name"]: item["id"] for item in RETAIL_CATALOG}
        basket_ids = [name_to_id[n] for n in active_cart_names if n in name_to_id]

        # Calculate Cart Value
        cart_total = sum(predictor.item_id_to_price.get(i, 0.0) for i in basket_ids)
        st.metric("Current Cart Total", f"₹{cart_total:.2f}", delta=f"{len(basket_ids)} Items")
    else:
        st.info("💡 **Next-Basket Mode:** Scoring all items purely against the user's historical graph embedding.", icon="💡")
        basket_ids = None
        active_cart_names = []

    # User History Accordion
    with st.expander(f"📜 View User #{user_id} Order History", expanded=False):
        user_history_ids = predictor.data["user_pos_map"].get(user_id, set())
        if user_history_ids:
            for item_id in sorted(user_history_ids):
                name = predictor.item_id_to_name.get(item_id, f"SKU #{item_id}")
                cat = predictor.item_id_to_category.get(item_id, "General")
                st.markdown(f"- **{name}** `({cat})`")
        else:
            st.caption("No historical orders recorded (cold start user).")

# -----------------------------------------------------------------------------
# Generate Recommendations
# -----------------------------------------------------------------------------
result = predictor.recommend(
    user_id=user_id,
    basket_item_ids=basket_ids,
    top_k=top_k,
    oos_skus=oos_skus,
    fefo_skus=fefo_skus if enable_fefo else None,
)

with col_right:
    st.subheader("✨ Top Complementary Recommendations")

    m_lat1, m_lat2, m_lat3 = st.columns(3)
    m_lat1.metric("Inference Latency", f"{result['inference_ms']} ms", help="Evaluated over full catalog graph")
    m_lat2.metric("OOS Masked", f"{result['filters_applied']['shelf_vision_oos_masked']} SKUs")
    m_lat3.metric("FEFO Boosted", f"{result['filters_applied']['fresh_grade_fefo_boosted']} SKUs")

    recs = result["recommendations"]

    if recs:
        for r in recs:
            rank = r["rank"]
            name = r["sku_name"]
            cat = r["category"]
            price = r["base_price_inr"]
            reason = r["reason"]
            score = r["score"]
            is_fefo = r["is_fefo_boosted"]

            fefo_badge = " 🍅 *[FEFO Clearance]*" if is_fefo else ""
            st.markdown(
                f"""
                <div class="rec-card">
                    <b>#{rank} {name}</b> ({cat}) — <b>₹{price:.2f}</b>{fefo_badge}<br>
                    <small style="color: #64748B;">💡 <i>{reason}</i> | Score: {score}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.warning("No recommendations passed active operational filters.")

# -----------------------------------------------------------------------------
# Economic Lift & JSON Payload
# -----------------------------------------------------------------------------
st.markdown("---")
c_exp1, c_exp2 = st.columns([0.65, 0.35])

with c_exp1:
    st.subheader("📦 Dark-Store Integration JSON Payload")
    st.caption("Standard contract dispatched to Quick-Commerce client carousel and picker routing totes.")

with c_exp2:
    st.download_button(
        "💾 Export Recommendations (JSON)",
        data=json.dumps(result, indent=2),
        file_name=f"basketgnn_user_{user_id}_recs.json",
        mime="application/json",
        use_container_width=True,
    )

with st.expander("🔍 View Raw JSON Payload", expanded=False):
    st.json(result)
