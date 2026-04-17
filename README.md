# Temporal Breast Cancer Risk Prediction
Comparison of temporal mammogram machine learning for breast cancer risk prediction with uncertainty quantified decision referrals 

## Abstract 
Breast cancer is one of the most common diseases and leading causes of mortality in women. Radiologists increasingly use AI systems as a second opinion to enhance screening accuracy. However, most deep learning models rely on a single screening exam and overlook longitudinal comparisons that are routinely used in clinical practice. A key challenge in using prior mammograms is that patient positioning and compression can make temporal comparisons unreliable without anatomical alignment. This paper conducts a comprehensive comparison of mammography screening models with multiple priors and spatial alignment. Using the publicly available Emory Breast Imaging Dataset (EMBED) Open Dataset, we trained models that predict the risk of breast cancer developing within one to five years by combining a patient’s current mammogram with all available prior scans and spatially aligning these scans taken at different time points. We compared five deep learning models to answer: (i) does temporal modeling improve future cancer risk prediction, and (ii) does spatial alignment improve future cancer risk prediction? We found that incorporating multiple prior scans did not consistently improve performance over using a single prior scan or the baseline model with no prior scans. Meanwhile, incorporating a single spatially aligned prior scan yielded a statistically significant improvement ($p < 0.05$) in AUPRC across all prediction horizons compared to using unaligned temporal priors. These results suggest that spatial alignment helps the model capture more meaningful longitudinal changes and, for mammography risk prediction, one aligned recent prior may be more informative than unaligned or multiple prior exams. Our findings provide guidance for designing future medical imaging systems that best leverage the available longitudinal information.

## Baseline Model Architecture 

<img width="740" height="294" alt="BaselineModel" src="https://github.com/user-attachments/assets/6094add2-13b8-49b9-a80a-2d2e93af6d13" />

The four mammographic views (L-CC, R-CC, L-MLO, and R-MLO) are passed through a shared ResNet-18 encoder to produce 512-dimensional feature representations. These features are fused using a view-attention pooling mechanism to obtain an exam-level representation, which is then processed by a multi-layer perceptron (MLP) head. A monotone cumulative hazard layer outputs multi-year breast cancer risk predictions.

## Temporal Model (Current + Most Recent Prior) Model Architecture 

<img width="836" height="359" alt="temporal1PriorArchitecture" src="https://github.com/user-attachments/assets/787a1273-a5b0-4223-8360-a2f82632a028" />

The current and prior mammographic views are encoded using a shared ResNet-18 backbone, aggregated through view-attention pooling, and combined with a temporal
difference representation. A transformer-based temporal module integrates these features before producing multi-year breast cancer risk predictions through a cumulative hazard layer.

## Temporal Model (Current + Multiple Prior Scans) Model Architecture 

<img width="810" height="360" alt="temporalMultiPriorArchitecture" src="https://github.com/user-attachments/assets/eb23c6ba-dff5-464e-8c57-d3f1c51abdbd" />

This shows the model architecture for the temporal model with the current mammogram and all available prior exams in the patient’s history. Visit-level features are extracted using a shared ResNet-18 and view-attention pooling, combined with temporal difference features and continuous time encoding, and processed by a transformer with visit-level attention before cumulative hazard prediction for 1–5 year risk.

## Temporal Model (Current + Most Recent Aligned Prior) Model Architecture 

<img width="837" height="360" alt="AlignedTemporalArchitecture" src="https://github.com/user-attachments/assets/950d7d88-3e17-4e49-bdc5-744bf2c8debe" />

This shows how the model uses aligned prior mammograms for input. The aligned and non-aligned temporal models share the same architecture, differing only in whether the
prior input is a raw prior mammogram or an aligned prior mammogram.

## Temporal Model (Current + Multiple Aligned Prior Scans) Model Architecture 

<img width="785" height="339" alt="multiPriorAlignedModelArch" src="https://github.com/user-attachments/assets/b2d371a1-e835-4a37-8870-98ecc54936ba" />

The aligned temporal model with multiple prior scans follows the same model architecture as the temporal model with the current and multiple prior scans in non- aligned setting with the only difference being the input of the aligned scans.

