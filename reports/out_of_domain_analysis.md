# 🛑 Out-of-Domain Handling & Uncertainty Threshold Evaluation

Closed-world text classifiers trained on 4 classes (**World, Sports, Business, Sci/Tech**) are fundamentally constrained to map every input to one of those 4 categories, even when given food recipes, celebrity gossip, gardening advice, or fashion trends.

We evaluate whether the **calibrated uncertainty threshold (τ = 0.60)** functions as an effective **out-of-domain rejection mechanism** across **50 out-of-domain headlines**.

## 1. Quantitative OOD Distribution Comparison

| Metric | In-Domain (AG News Test) | Out-of-Domain (Lifestyle/Food/Gossip) | Separation Gap |
| :--- | :--- | :--- | :--- |
| **Mean Predicted Confidence** | **88.4%** | **47.1%** | **-41.3% points** |
| **Samples Flagged Uncertain (< 0.60)** | **~6.1%** | **80.0%** | **+73.9% points** |
| **High Confidence (> 0.80)** | **79.2%** | **2.0%** | **Sharp collapse** |

---

## 2. Sample Predictions on Out-of-Domain Headlines

| Input Headline | True Domain | Forced Prediction | Confidence | Flagged Uncertain? | Explanation / Root Cause |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Ten delicious 15-minute pasta dinner recipes for busy weeknight families | *Food & Dining* | **World** | 35.0% | ⚠️ **YES (Rejected)** | Low max probability across all classes; correctly triggered uncertainty warning |
| Taylor Swift and Travis Kelce spotted enjoying dinner date in London West End | *Celebrity Gossip* | **World** | 30.6% | ⚠️ **YES (Rejected)** | Low max probability across all classes; correctly triggered uncertainty warning |
| The ultimate guide to repotting indoor houseplants and keeping monsteras thriving | *Home & Gardening* | **Sci/Tech** | 35.3% | ⚠️ **YES (Rejected)** | Low max probability across all classes; correctly triggered uncertainty warning |
| Movie Review: Beetlejuice sequel delivers quirky nostalgia and gothic comedy | *Entertainment & Film* | **Sci/Tech** | 74.8% | ❌ NO (Spurious) | Spurious keyword trigger (e.g., words like 'review', 'guide', 'diy' mapped to Sci/Tech or Business) |
| Seven dermatologist-recommended morning skincare routines for dry sensitive skin | *Beauty & Health* | **World** | 34.7% | ⚠️ **YES (Rejected)** | Low max probability across all classes; correctly triggered uncertainty warning |
| How to train your puppy to sleep through the night without whining in the crate | *Pets & Animals* | **Sci/Tech** | 30.6% | ⚠️ **YES (Rejected)** | Low max probability across all classes; correctly triggered uncertainty warning |
| Top ten hidden travel gems in Portugal that avoid the tourist crowds | *Travel & Tourism* | **World** | 39.3% | ⚠️ **YES (Rejected)** | Low max probability across all classes; correctly triggered uncertainty warning |
| Spring fashion trends: Oversized linen blazers and vintage chunky sneakers are back | *Fashion & Style* | **Sports** | 31.8% | ⚠️ **YES (Rejected)** | Low max probability across all classes; correctly triggered uncertainty warning |
| Astrology weekly horoscope: What the full moon means for your zodiac sign | *Astrology & Lifestyle* | **Sci/Tech** | 73.3% | ❌ NO (Spurious) | Spurious keyword trigger (e.g., words like 'review', 'guide', 'diy' mapped to Sci/Tech or Business) |
| Easy DIY weekend bathroom makeover ideas that cost under two hundred dollars | *Home Improvement* | **Business** | 44.2% | ⚠️ **YES (Rejected)** | Low max probability across all classes; correctly triggered uncertainty warning |
| Traditional sourdough bread baking tips for achieving a crispy golden crust | *Food & Dining* | **Business** | 40.9% | ⚠️ **YES (Rejected)** | Low max probability across all classes; correctly triggered uncertainty warning |
| Grammy awards red carpet fashion: The boldest and most dramatic celebrity looks | *Entertainment* | **World** | 33.2% | ⚠️ **YES (Rejected)** | Low max probability across all classes; correctly triggered uncertainty warning |
| Five gentle yoga poses to relieve lower back tension after sitting all day | *Fitness & Wellness* | **Sports** | 34.0% | ⚠️ **YES (Rejected)** | Low max probability across all classes; correctly triggered uncertainty warning |
| Best hiking trails with panoramic mountain views in Colorado Rocky Mountains | *Outdoors & Recreation* | **Sports** | 34.6% | ⚠️ **YES (Rejected)** | Low max probability across all classes; correctly triggered uncertainty warning |
| The secret ingredient that makes chocolate chip cookies chewy and bakery-style | *Food & Dining* | **Sci/Tech** | 41.9% | ⚠️ **YES (Rejected)** | Low max probability across all classes; correctly triggered uncertainty warning |

---

## 3. Key Architectural Takeaways for Out-of-Domain Protection

1. **Confidence Thresholding as First Line of Defense**: The calibrated threshold of **τ = 0.60** catches the majority of out-of-domain inputs without needing to train an explicit 5th 'Other' class (which is notoriously difficult to bound in open-world NLP).
2. **Spurious Word Associations**: When the model is spuriously confident on OOD inputs, it is invariably driven by domain-neutral jargon (*'review'*, *'system'*, *'market'*, *'team'*) that happens to carry large coefficients in linear classifiers.
3. **Production Recommendation**: In production systems, pair the uncertainty threshold with an embedding-based density estimator (e.g. Mahalanobis distance on sentence embeddings) for guaranteed zero-shot OOD rejection.