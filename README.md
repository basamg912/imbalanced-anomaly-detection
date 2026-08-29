# Imbalanced Anomaly Detector

**Anomaly Detection on Highly Imbalanced Data with SMOTE, Cost-Sensitive Learning & Ensemble Methods**

정상 데이터가 비정상 데이터보다 크게 많은 **불균형 데이터셋에서 이상 탐지 알고리즘의 성능을 비교·분석**한 개인 프로젝트다.

제조업 이상 탐지 문제에서 영감을 받아 **Wine Quality Dataset**을 이상 탐지 문제로 재구성하고, 다양한 분류 및 이상치 탐지 방법을 비교했다.

특히 단순 Accuracy 대신 **Equal Error Rate (EER)**를 주요 평가 지표로 사용하여 클래스 불균형 상황에서 모델의 탐지 성능을 비교했다.

## Dataset

**Wine Quality Dataset (Red Wine)**을 다음과 같이 이진 이상 탐지 문제로 재구성했다.

| Class       | Definition  | Samples |
| ----------- | ----------- | ------: |
| Normal (0)  | Quality ≤ 6 |   1,382 |
| Anomaly (1) | Quality > 6 |     217 |

전체 데이터에서 정상 데이터가 약 **85%**, 비정상 데이터가 약 **15%**를 차지하여 약 **85:15의 class imbalance**가 발생한다.

```text id="p1m8q3"
Wine Quality Dataset
        │
        ▼
Quality Score
        │
   ┌────┴────┐
   ▼         ▼
 ≤ 6        > 6
   │         │
Normal     Anomaly
 85%         15%
```

## Problem

불균형 데이터에서는 Accuracy가 모델 성능을 제대로 반영하지 못할 수 있다.

예를 들어 모든 샘플을 정상으로 예측하더라도 약 86%의 Accuracy를 얻을 수 있다. 따라서 **False Acceptance와 False Rejection의 균형**을 고려할 수 있는 EER을 주요 평가 지표로 사용했다.

## Evaluation — Equal Error Rate

**Equal Error Rate (EER)**는 분류 임계값을 변화시켰을 때 **False Acceptance Rate (FAR)**와 **False Rejection Rate (FRR)**가 같아지는 지점의 오류율이다.

낮은 EER일수록 정상과 이상 데이터를 더 안정적으로 구분하는 모델로 해석할 수 있다.

```text id="v4q8k2"
             Model Score
                  │
                  ▼
          Threshold Sweep
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
       FAR                 FRR
        │                   │
        └─────────┬─────────┘
                  ▼
                 EER
```

## Methods

불균형 문제와 이상 탐지 문제에 대응하기 위해 여러 접근법을 비교했다.

### Distance-based Methods

* K-Nearest Neighbors (KNN)
* Weighted KNN

거리 기반 방법을 통해 데이터 공간에서 주변 샘플과의 관계를 이용하여 이상 여부를 판단했다.

### Anomaly Detection Methods

* One-Class SVM
* ABOD (Angle-Based Outlier Detection)
* Isolation Forest

비지도 또는 반지도 방식의 전용 이상치 탐지 알고리즘을 적용하여 성능을 비교했다.

### Imbalance Handling

불균형 데이터에 대한 대표적인 접근법을 적용하여 모델 학습 과정에서의 class imbalance 문제를 완화했다.

* SMOTE
* Cost-Sensitive Learning
* Gradient Boosting Ensemble

## Experiment Pipeline

```text id="n7q2w9"
Wine Quality Dataset
        ↓
Binary Anomaly Labeling
        ↓
Highly Imbalanced Dataset
        ↓
┌──────────────────────────────┐
│     Imbalance Handling       │
│   SMOTE / Cost-Sensitive     │
└──────────────┬───────────────┘
               ↓
      Multiple Algorithms
               │
    ┌──────────┼──────────┐
    ↓          ↓          ↓
   KNN       OCSVM     Isolation
   WKNN       ABOD       Forest
    │          │          │
    └──────────┼──────────┘
               ↓
          EER Evaluation
               ↓
       Model Comparison
```

## Results

실험 결과, 이 데이터셋에서는 **거리 기반 방법이 가장 우수한 성능**을 보였다.

### Key Findings

**Weighted KNN**이 가장 좋은 성능을 기록했으며, 지역적인 거리 정보를 가중치로 반영하면서 **EER을 0.20 이하**로 낮췄다.

반면 **One-Class SVM과 ABOD**는 이 데이터셋에서 상대적으로 낮은 성능을 보였다.

**Isolation Forest**는 **EER = 0.4432**를 기록하며 비교적 안정적인 baseline으로 동작했다.

```text id="6yq4bz"
Performance Trend

Weighted KNN
     ↓
 EER < 0.20
     ↓
 Best Performance

Isolation Forest
     ↓
 EER = 0.4432
     ↓
 Robust Baseline

One-Class SVM / ABOD
     ↓
 Relatively Lower Performance
```

## Conclusion

이상 탐지에서는 모델의 복잡도나 특정 알고리즘의 일반적인 성능만으로 결과를 예측하기 어렵다.

이번 실험에서는 전용 이상치 탐지 알고리즘보다 **단순한 거리 기반 방법인 KNN 계열이 더 우수한 성능**을 보였다.

이를 통해 이상 탐지 모델의 선택은 알고리즘 자체의 복잡성보다 **데이터의 분포와 feature space의 구조에 맞춰 실험적으로 검증하는 것이 중요하다**는 점을 확인했다.

## Tech Stack

| Component          | Technology                                               |
| ------------------ | -------------------------------------------------------- |
| Dataset            | Wine Quality Dataset                                     |
| Anomaly Detection  | KNN, Weighted KNN, One-Class SVM, ABOD, Isolation Forest |
| Imbalance Handling | SMOTE, Cost-Sensitive Learning                           |
| Ensemble           | Gradient Boosting                                        |
| Evaluation         | Equal Error Rate (EER)                                   |
| Language           | Python                                                   |
