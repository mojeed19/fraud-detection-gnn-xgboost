# Project Topics: Cost‑Sensitive Fraud Detection with Graph Neural Networks
## Abstract
A cost‑sensitive fraud detection system was developed using a Graph Neural Network (GraphSAGE) and an XGBoost baseline on a balanced credit card transaction dataset (fraud rate 50%). The GraphSAGE model achieved an Average Precision (AP) of 0.9724, exceeding the target of 0.95, but showed poor calibration with an Expected Calibration Error (ECE) of 0.3009. In contrast, the XGBoost model delivered near‑perfect discrimination (AP 0.9998) and was already well‑calibrated (ECE 0.0096). Isotonic calibration further improved its ECE to 0.0018, well below the 0.05 target. Cost‑sensitive threshold optimisation, with false‑negative cost $500 and false‑positive cost $10, reduced expected financial loss by 87.8% (from $4,190 to $510) by lowering the optimal decision threshold to 0.120. SHAP analysis revealed that most features contributed a strong negative signal, pushing predictions toward the legitimate class. The calibrated XGBoost model is recommended for deployment due to its excellent discrimination, reliable probability estimates, and significant cost savings.

## Keywords

Fraud detection, Graph Neural Networks, GraphSAGE, XGBoost, cost‑sensitive learning, isotonic calibration, expected calibration error, SHAP, financial risk.

## Data Source

The dataset used is **Credit Card Fraud Detection Dataset 2023** (`creditcard_2023.csv`). It contains 568,630 transactions with 29 anonymised features (V1–V28) plus the `Amount` and `Class` (target) columns. The fraud rate is exactly 50% – a balanced design to facilitate model training. The dataset was obtained from a publicly available source (e.g., Kaggle). No missing values were present; the `id` column was dropped as non‑informative.

## Proving Mathematical Methodology

### 1. Graph Construction

An undirected graph \( G = (V, E) \) is built from a subsampled set of 20,000 transactions (stratified to preserve 50% fraud rate). Each node \( v_i \) represents a transaction with feature vector \( \mathbf{x}_i \in \mathbb{R}^{29} \). Edges are added using \( k \)-nearest neighbours (\( k=5 \)) with cosine distance:

\[
\text{cosine distance}(\mathbf{x}_i, \mathbf{x}_j) = 1 - \frac{\mathbf{x}_i \cdot \mathbf{x}_j}{\|\mathbf{x}_i\|_2 \|\mathbf{x}_j\|_2}.
\]

The resulting adjacency matrix \( \mathbf{A} \) is stored as a sparse tensor. The graph has 20,000 nodes and 100,000 edges (20,000 × 5, but each edge counted once).

### 2. GraphSAGE Model

A two‑layer GraphSAGE network (Hamilton et al., 2017) is used. The propagation rule at layer \( \ell \) is:

\[
\mathbf{h}_i^{(\ell+1)} = \sigma \left( \mathbf{W}_1^{(\ell)} \mathbf{h}_i^{(\ell)} + \mathbf{W}_2^{(\ell)} \cdot \text{mean}_{j \in \mathcal{N}(i)} \mathbf{h}_j^{(\ell)} \right),
\]

where \( \mathbf{h}_i^{(\ell)} \) is the hidden representation of node \( i \), \( \mathcal{N}(i) \) its neighbours, \( \sigma \) the ReLU activation, and \( \mathbf{W}_1^{(\ell)}, \mathbf{W}_2^{(\ell)} \) learnable weight matrices. The final layer outputs a 2‑dimensional vector passed through softmax to obtain fraud probability \( \hat{p}_i = P(y_i = 1 \mid \mathbf{x}_i, G) \).

### 3. Cost‑Sensitive Loss

Weighted binary cross‑entropy penalises false negatives 50× more than false positives:

\[
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[ w_{y_i} \cdot \big( y_i \log \hat{p}_i + (1-y_i) \log (1-\hat{p}_i) \big) \right],
\]

with \( w_0 = 1 \) (cost of false positive) and \( w_1 = 50 \) (cost of false negative).

### 4. Isotonic Calibration (XGBoost Baseline)

Raw probabilities \( \hat{p}_i^{\text{(raw)}} \) from XGBoost are calibrated using isotonic regression on a held‑out calibration set (20% of training data). The calibrated probability is:

\[
\hat{p}_i^{\text{(cal)}} = \phi(\hat{p}_i^{\text{(raw)}}), \quad \phi \text{ monotone non‑decreasing},
\]

where \( \phi \) minimises squared error \( \sum_i (\phi(\hat{p}_i^{\text{(raw)}}) - y_i)^2 \) under the monotonicity constraint.

### 5. Expected Calibration Error (ECE)

The probability range \([0,1]\) is split into \( M=10 \) equal‑frequency bins \( B_m \). For each bin, compute accuracy and confidence:

\[
\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \mathbf{1}_{y_i = 1}, \qquad
\text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_i.
\]

Then

\[
\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|.
\]

A perfectly calibrated model has ECE = 0; target ≤ 0.05.

### 6. Cost‑Optimal Threshold

For a decision threshold \( \tau \), predict fraud if \( \hat{p}_i \ge \tau \). Expected total cost on the test set:

\[
R(\tau) = C_{FN} \cdot FN(\tau) + C_{FP} \cdot FP(\tau),
\]

with \( C_{FN} = \$500 \) (missed fraud), \( C_{FP} = \$10 \) (false alarm). The optimal threshold \( \tau^* = \arg\min_{\tau} R(\tau) \) is found by grid search over 101 points.

### 7. Average Precision (AP)

AP is the area under the precision‑recall curve, computed as:

\[
\text{AP} = \sum_{k=1}^{K} (R_k - R_{k-1}) P_k,
\]

where \( P_k \) and \( R_k \) are precision and recall at threshold \( \tau_k \). Target AP ≥ 0.95.

### 8. SHAP Explainability

For the XGBoost model, SHAP (SHapley Additive exPlanations) provides additive feature attribution:

\[
\hat{p}_i = \phi_0 + \sum_{j=1}^{d} \phi_j z_{ij},
\]

where \( \phi_j \) is the contribution of feature \( j \) to the prediction. SHAP values satisfy local accuracy, consistency, and missingness.

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
