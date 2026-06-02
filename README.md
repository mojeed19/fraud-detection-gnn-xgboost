# Project Topics: Cost‑Sensitive Fraud Detection with Graph Neural Networks
## Abstract
A cost‑sensitive fraud detection system was developed using a Graph Neural Network (GraphSAGE) and an XGBoost baseline on a balanced credit card transaction dataset (fraud rate 50%). The GraphSAGE model achieved an Average Precision (AP) of 0.9724, exceeding the target of 0.95, but showed poor calibration with an Expected Calibration Error (ECE) of 0.3009. In contrast, the XGBoost model delivered near‑perfect discrimination (AP 0.9998) and was already well‑calibrated (ECE 0.0096). Isotonic calibration further improved its ECE to 0.0018, well below the 0.05 target. Cost‑sensitive threshold optimisation, with false‑negative cost $500 and false‑positive cost $10, reduced expected financial loss by 87.8% (from $4,190 to $510) by lowering the optimal decision threshold to 0.120. SHAP analysis revealed that most features contributed a strong negative signal, pushing predictions toward the legitimate class. The calibrated XGBoost model is recommended for deployment due to its excellent discrimination, reliable probability estimates, and significant cost savings.

## Keywords

Fraud detection, Graph Neural Networks, GraphSAGE, XGBoost, cost‑sensitive learning, isotonic calibration, expected calibration error, SHAP, financial risk.
## Aim

To develop a cost‑sensitive, uncertainty‑aware fraud detection system using Graph Neural Networks (GNNs) that captures relational patterns among financial transactions, quantifies predictive uncertainty, and optimises decision thresholds to minimise expected financial loss, while providing explainable predictions for banking compliance.

## Objectives

1. **Construct a transaction graph** from the Credit Card Fraud Detection Dataset 2023 (creditcard_2023.csv) using k‑nearest neighbours (k‑NN) in the feature space, linking transactions with similar characteristics.  
2. **Build a GraphSAGE model** (a type of GNN) that propagates information across the graph to learn node‑level fraud probabilities.  
3. **Incorporate a cost‑sensitive loss function** (weighted binary cross‑entropy) where false negatives (missed fraud) are penalised 50× more than false positives (legitimate transactions flagged as fraud).  
4. **Quantify epistemic uncertainty** via the model’s output variance (not a full ensemble; instead we rely on the inherent variability of the GNN).  
5. **Calibrate the XGBoost baseline** using isotonic regression to improve probability reliability (Expected Calibration Error ≤ 0.05).  
6. **Optimise the decision threshold** to minimise expected cost with \( C_{FN} = \$500 \), \( C_{FP} = \$10 \).  
7. **Evaluate using Average Precision (AP)** (target ≥ 0.95), **Expected Calibration Error (ECE)** (target ≤ 0.05), **Brier score**, and **cost saving**.  
8. **Provide explainability** using SHAP values on the XGBoost baseline to identify the most influential features for each transaction.


## Data Source

The dataset used is **Credit Card Fraud Detection Dataset 2023** (`creditcard_2023.csv`). It contains 568,630 transactions with 29 anonymised features (V1–V28) plus the `Amount` and `Class` (target) columns. The fraud rate is exactly 50% – a balanced design to facilitate model training. The dataset was obtained from a publicly available source (e.g., Kaggle). No missing values were present; the `id` column was dropped as non‑informative.

## Proving Mathematical Methodology
<img width="910" height="413" alt="image" src="https://github.com/user-attachments/assets/378bad1b-9eff-467f-8c81-f375d5807eab" />
<img width="986" height="496" alt="image" src="https://github.com/user-attachments/assets/9f2ea3a4-4dd6-438b-a833-17dd2c6926ee" />
<img width="1003" height="458" alt="image" src="https://github.com/user-attachments/assets/2a265da9-e6a7-4e9f-9c79-96d56dd9d2f7" />
<img width="995" height="483" alt="image" src="https://github.com/user-attachments/assets/08069689-da07-4197-8c92-c4473027fba7" />

## Key Insight

- **GraphSAGE learns meaningful patterns** (AP 0.9724) but suffers from **severe miscalibration** (ECE 0.3009). This indicates that while the graph structure helps discriminate fraud, the output probabilities are not trustworthy as risks.
- **XGBoost already achieves near‑perfect discrimination** (AP 0.9998) and is **initially well‑calibrated** (ECE 0.0096). Isotonic calibration further reduces ECE to 0.0018, making probability estimates extremely reliable.
- **Cost‑sensitive threshold tuning is critical**: lowering the threshold from 0.5 to 0.120 reduces expected cost by 87.8%, highlighting that the business cost asymmetry (FN cost 50× FP cost) strongly favours a more sensitive model.
- **SHAP analysis** shows that most features (V14, V4, V12, etc.) have a uniform negative contribution (≈ -3.8), indicating a strong baseline push toward “legitimate”. Only a few features occasionally provide positive contributions.

## Result and Finding

### GraphSAGE Model
| Metric | Value | Target |
|--------|-------|--------|
| Average Precision (AP) | 0.9724 | ≥0.95 |
| Expected Calibration Error (ECE) | 0.3009 | ≤0.05 |
| Brier Score | 0.1707 | – |

GraphSAGE meets the AP target but fails calibration.

### XGBoost Model (Raw vs Calibrated)
| Metric | Raw | Calibrated | Target |
|--------|------|------------|--------|
| AP | 0.9998 | 0.9996 | ≥0.95 |
| ECE | 0.0096 | 0.0018 | ≤0.05 |
| Calibration improvement | – | 0.0078 | – |

XGBoost exceeds both AP and ECE targets. Calibration improves ECE further.

### Cost Analysis (Calibrated XGBoost)
| Threshold | Expected Cost | Saving |
|-----------|---------------|--------|
| 0.5 (conventional) | $4,190 | – |
| 0.120 (optimal) | $510 | $3,680 (87.8%) |

### Reliability Diagram
After isotonic calibration, the model’s predicted probabilities align perfectly with observed fractions in the 0.04–0.12 range, and show excellent agreement across the whole probability spectrum.

### SHAP Summary
The SHAP summary plot (Figure in notebook) shows that nearly all features push the model toward a negative base value of –3.8, indicating that the model starts with a strong prior for “legitimate”. A few features occasionally exhibit positive SHAP values, increasing the fraud probability.

## Recommendation

1. **Deploy the calibrated XGBoost model** for real‑time fraud scoring. It provides both excellent discrimination (AP 0.9996) and reliable probability estimates (ECE 0.0018), which are essential for risk‑based decision making and regulatory compliance.
2. **Use the cost‑optimal threshold of 0.120** rather than the default 0.5. This reduces expected financial loss by nearly 88%.
3. **Monitor calibration drift** over time using ECE on a rolling basis; retrain the isotonic calibrator periodically.
4. **Incorporate SHAP explanations** into the fraud review dashboard to help investigators understand why a transaction received a high fraud score.
5. **GraphSAGE is not recommended** for production due to poor calibration, although the graph construction method could be used to generate additional features for XGBoost.

## Conclusion

This project successfully demonstrated a cost‑sensitive fraud detection pipeline using both a Graph Neural Network and a gradient‑boosted tree baseline. While GraphSAGE achieved acceptable discrimination, its probability estimates were poorly calibrated, making it unsuitable for risk‑sensitive applications. XGBoost, on the other hand, delivered outstanding discrimination and calibration, and when combined with isotonic calibration and cost‑optimal thresholding, reduced expected financial loss by 87.8%. SHAP analysis provided transparency into the model’s decisions. The final recommendation is to deploy the calibrated XGBoost model with a threshold of 0.120 to maximise fraud prevention while minimising operational costs.

## Reference

- Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive representation learning on large graphs. *Advances in Neural Information Processing Systems (NIPS)*.
- Naeini, M. P., Cooper, G. F., & Hauskrecht, M. (2015). Obtaining well calibrated probabilities using Bayesian binning. *AAAI Conference on Artificial Intelligence*.
- Elkan, C. (2001). The foundations of cost‑sensitive learning. *International Joint Conference on Artificial Intelligence (IJCAI)*.
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems (NIPS)*.
- Credit Card Fraud Detection Dataset 2023. (2023). Available at: https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023
