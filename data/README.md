# BasketGNN Datasets & Benchmarks

BasketGNN supports three primary data sources:

---

## 1. Built-in Quick-Commerce Dark Store Generator (Default)

The project includes a built-in generator modeling real Milkbasket order topologies (dairy, bakery, perishables, staples, and home care). It automatically synthesizes co-purchase clusters (e.g. Milk + Bread + Eggs + Butter, Tea + Biscuits, Cooking Staples).

To regenerate:
```bash
python src/data.py
```
This generates `interactions.csv`, `catalog.json`, and `metadata.json` under `data/processed/`.

---

## 2. Instacart Market Basket Analysis (Kaggle)

* **Dataset URL:** [kaggle.com/c/instacart-market-basket-analysis](https://www.kaggle.com/c/instacart-market-basket-analysis)
* **Scale:** ~3 million grocery orders, 206,209 users, 49,688 products.
* **Setup:**
  1. Download `order_products__prior.csv.zip`, `orders.csv.zip`, and `products.csv.zip`.
  2. Unzip into `data/raw/instacart/`.
  3. Preprocess with:
     ```bash
     python src/data.py --source data/raw/instacart --source-format instacart --out data/processed
     ```

---

## 3. Dunnhumby "The Complete Journey"

* **Dataset URL:** [dunnhumby.com/source-files](https://www.dunnhumby.com/source-files/)
* **Scale:** ~2,500 households over 2 continuous years of supermarket transaction logs.
* **Setup:** Place `transaction_data.csv` and `product.csv` into `data/raw/dunnhumby/`.
