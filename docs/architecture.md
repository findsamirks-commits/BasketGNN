# BasketGNN Architecture & Design Rationale

## 1. Why LightGCN for Quick-Commerce?

Traditional Graph Convolutional Networks (GCNs) and Neural Graph Collaborative Filtering (NGCF) borrow feature transformations (weight matrices $W$) and non-linear activations ($\text{ReLU}, \sigma$) directly from computer vision.

However, He et al. (SIGIR 2020) demonstrated that in collaborative filtering:
- Nodes do not possess high-dimensional sensory features (only discrete IDs).
- Heavy non-linear transformations complicate gradient flow and degrade ranking performance.
- **Pure linear neighborhood propagation** captures the essential signal: high-order graph connectivity.

### Mathematical Formulation
Given the symmetric-normalized bipartite adjacency matrix:
$$\tilde{A} = D^{-1/2} A D^{-1/2}$$

Embeddings propagate across $K$ layers:
$$E^{(k+1)} = \tilde{A} E^{(k)}$$

The final representation combines multi-hop graph embeddings:
$$E = \sum_{k=0}^K \alpha_k E^{(k)}, \quad \alpha_k = \frac{1}{K + 1}$$

---

## 2. Dual-Head Prediction Modes

1. **User Query (Next-Basket Prediction):**
   $$\hat{y}_{ui} = e_u \cdot e_i^T$$
   Used for app homepages, daily push notifications, and subscription replenishment.

2. **Cart Query (Basket Completion & Cross-Sell):**
   $$q = e_u + \alpha \cdot \frac{1}{|B|} \sum_{i \in B} e_i$$
   $$\hat{y}_{qi} = q \cdot e_i^T$$
   Used in checkout bottom-sheets, cart sliders, and "Frequently Bought Together" widgets.

---

## 3. Operational Integration: Closed-Loop Mesh

```
┌──────────────────┐     ┌──────────────┐     ┌────────────┐     ┌──────────────┐
│ Shelf-Vision     │────▶│ BasketGNN    │◀────│ Fresh-Grade│────▶│ Perishable-  │
│ (OOS detection)  │ OOS │ (recommend)  │SELL │ (grade+days)│     │ Markdown-    │
│                  │mask │              │FIRST│            │     │ Optimizer    │
└──────────────────┘     └──────┬───────┘     └────────────┘     └──────────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │ SASRec / Two-Tower   │
                     │ as ensemble members  │
                     └──────────────────────┘
```

* **Shelf-Vision OOS Masking:** Adds a $-100.0$ score penalty to any item detected out of stock on dark store shelves, completely eliminating ghost recommendations.
* **Fresh-Grade FEFO Boosting:** Adds a commercial priority bonus ($\text{boost} = w \cdot (1 + 1/d)$) to perishable items in `SELL_FIRST` status, clearing near-expiry produce before markdown erosion.
