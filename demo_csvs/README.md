# Demo CSV Samples for Batch Prediction Testing

This directory contains pre-built, realistic CSV datasets designed to test the **Batch CSV Prediction** interface of the News Category Classifier.

---

## Available Test Files

| File | Rows | Main Text Column | Description & What to Test |
|:-----|:-----|:-----------------|:---------------------------|
| [`general_news_batch.csv`](file:///c:/Users/Devendra/projects/News_Category_Classifier/demo_csvs/general_news_batch.csv) | 20 | `headline` | **Balanced Benchmark:** Contains 5 representative headlines for each of the 4 categories (World, Sports, Business, Sci/Tech). Tests standard classification throughput. |
| [`sports_and_athletics.csv`](file:///c:/Users/Devendra/projects/News_Category_Classifier/demo_csvs/sports_and_athletics.csv) | 15 | `news_text` | **Sports Focus & Alternative Column Name:** Tests automatic detection of `news_text` column and verifies high-confidence Sports predictions across soccer, tennis, basketball, F1, and rugby. |
| [`business_and_markets.csv`](file:///c:/Users/Devendra/projects/News_Category_Classifier/demo_csvs/business_and_markets.csv) | 12 | `title` | **Multi-Column Dataset:** Contains `id`, `ticker`, `title`, and `market_date`. Tests the column selector dropdown to choose `title` among multiple metadata columns. |
| [`science_and_technology.csv`](file:///c:/Users/Devendra/projects/News_Category_Classifier/demo_csvs/science_and_technology.csv) | 10 | `content` | **Sci/Tech Focus & Content Column:** Contains `article_id`, `category_hint`, `content`, `reading_time_mins`. Tests selecting `content` and verifies quantum computing, AI, space, and biotech classification. |
| [`world_diplomacy_and_affairs.csv`](file:///c:/Users/Devendra/projects/News_Category_Classifier/demo_csvs/world_diplomacy_and_affairs.csv) | 10 | `headline` | **World & Geopolitics Focus:** Diplomatic summits, UN resolutions, multilateral pacts, and international treaties. |
| [`challenging_and_edge_cases.csv`](file:///c:/Users/Devendra/projects/News_Category_Classifier/demo_csvs/challenging_and_edge_cases.csv) | 12 | `headline` | **Uncertainty & Reliability Stress Test:** Tests very short headlines (2 words), multi-domain crossover headlines (Sports-Business acquisitions, State-sponsored cyber warfare), ambiguous phrases, and out-of-domain recipes/dog training. Verify that the UI flags low confidence with `Low Confidence` status. |

---

## How to Test in the Web UI

1. Open the dashboard at `http://127.0.0.1:5050/app/` (or via Streamlit in tab 2).
2. Click on the **Batch CSV Prediction** tab.
3. Drag & drop any `.csv` file from this folder into the dropzone (or click **browse from your computer**).
4. Verify that:
   - The filename, detected row count, and file size are accurately displayed.
   - The correct text column is automatically detected or selectable via the dropdown.
5. Click **Run Batch Prediction**.
6. Review the results table with predicted categories, confidence percentages, and status tags (`Confident` vs `Low Confidence`).
7. Click **Download Results CSV** to export the classified output.
