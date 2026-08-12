# TRE ≈ 0 수렴 근거 정리

> tre-diffusion-image-detection 재현 실험 중 확인된 핵심 발견.
> 나중에 리포 `docs/`에 포함하거나 README Notes로 요약 예정.

## 주장

원본 구현(노트북 및 리포 `src/data/`)의 TRE(Temporal Reconstruction Error)
특징은 **이론적으로 정확히 0**이며, 실제 저장되는 값은 GPU 부동소수점
비결정성 잔차다. 따라서 분류기는 의도된 "시간적 재구성 오차"가 아니라
수치 잔차의 패턴을 학습한다.

## 근거 1 — 수학적: edit-friendly 인버전의 정확 복원 성질

구현이 사용하는 인버전은 edit-friendly DDPM inversion
(Huberman-Spiegelglas et al.)이다. 순방향에서 노이즈 latent 열
`x_1..x_T`를 샘플링한 뒤(`sample_xts_from_x0`), 각 스텝의 노이즈를

```
z_t = (x_{t-1} - mu_t(x_t)) / sigma_t        # inversion_forward_process
```

로 **역산해 저장**한다. 역과정(`inversion_reverse_process` +
`reverse_step`)은 같은 `mu_t`, `sigma_t` 공식으로

```
x_{t-1} = mu_t(x_t) + sigma_t * z_t
```

를 계산하므로, 저장된 `z_t`를 그대로 재주입하면 **구성상 x_{t-1}이 정확히
재현**된다. 귀납적으로, 어떤 스텝 k의 `x_k`에서 출발하든 재구성 결과는
정확히 `x_0`(원본 latent)이다.

TRE는 "prefix k로 복원한 latent"와 "prefix k+1로 복원한 latent"의 차이인데
(`over_denosing`, `LoadDataset.ipynb` cell 6 = 리포 `tre_features.py`),
위 성질에 의해 **모든 prefix 복원이 동일한 x_0**이므로 차이는 0이다.

(유일한 예외: 구현이 `zs[0] = 0`으로 마지막 스텝 노이즈를 지우는데, 이
편차는 모든 prefix 복원에 공통으로 들어가므로 차이에서 상쇄된다.)

## 근거 2 — 실측: 서버 재현에서의 통계

GPU 서버(L40S, fp32, torch 2.3.1+cu121, diffusers 0.31.0)에서 원본 코드
경로(`src.data.tre_features.compute_tre`)로 실측:

```
TRE shape: (20, 4, 32, 32), dtype float32
mean ≈ 4.5e-14,  std ≈ 2.7e-4
```

SD latent의 자연 스케일이 O(1)임을 감안하면 신호 크기가 약 3~4 자릿수
작다. 0이 아닌 이유는 UNet 추론의 커널 비결정성(같은 입력이라도 실행마다
미세하게 다른 부동소수점 결과) 때문이며, 이는 순방향(z 역산 시)과
역방향(재주입 시)의 UNet 출력이 비트 단위로 일치하지 않아 생긴다.

## 근거 3 — 원본 노트북과의 대조

- `LoadDataset.ipynb` cell 6의 `over_denosing`은 리포 코드와 동일하게
  `inversion_reverse_process(..., zs=zs[:step+1])`로 **z_t를 재주입**한다.
  즉 이 성질은 리팩터링으로 생긴 것이 아니라 원본부터 존재했다.
- 같은 노트북의 분류기 학습 셀(cell 16)은 `temporal_module`이 정의되지
  않아 실행 불가능한 상태로 남아 있다 — 실험이 완주되지 않았음을 시사.

## 성능 수치에 대한 주의

원논문 보고 수치(Ours_TRE 66.0% 등)는 실험이 제대로 완료되지 않은
상태에서 나온 것으로 **신뢰할 수 없다**(저자 확인). 따라서 본 재현에서는
논문 수치를 재현 목표나 비교 기준으로 삼지 않으며, 이번 전체 규모
재실행(학습 60k / 평가 96k, 프로토콜 문서화)이 **이 파이프라인의 첫 유효
측정치**가 된다.

## 재현 실측 결과 (전체 규모)

학습 sdv1.4 30k+30k(seed 42) / 평가 8개 생성기 전량, 20 epochs.
분류기·프로토콜은 원본 노트북 레시피 그대로. 원본 수치는 `results/repro.json`.

| Generator | Accuracy | AP |
| --- | --- | --- |
| sdv4 (학습 도메인) | 61.5% | 0.642 |
| sdv5 | 62.2% | 0.648 |
| wukong | 59.8% | 0.623 |
| glide | 58.1% | 0.593 |
| vqdm | 56.2% | 0.562 |
| midjourney | 55.0% | 0.552 |
| biggan | 54.7% | 0.560 |
| adm | 54.4% | 0.541 |
| **평균** | **57.7%** | — |

해석: **학습 도메인에서조차 61.5%**에 머물고, 보지 못한 생성기에서는
54~60%(우연 수준 50%에 근접)로 떨어진다. 즉 z 재주입 TRE에는 실질적인
판별 신호가 거의 없고, 남은 약한 신호는 본 문서가 분석한 수치 잔차의
이미지 의존적 패턴으로 설명된다. 학습 과정도 이와 일관되게 val acc가
52~61%에서 진동했다.

(참고: 초기 sanity 실험에서 val acc 79.5%가 관측됐으나, 당시 데이터가
fake 79% 불균형이어서 다수 클래스 예측에 수렴한 착시였다 — val acc가
fake 비율과 일치.)

## 개선 실험 설계 (fresh-noise TRE)

재주입 대신 **미리 뽑은 공통 난수 열 eps_t를 모든 prefix 복원에 공유**하여
재구성한다:

- prefix별 복원이 실제로 달라져 TRE가 "모델이 이미지를 얼마나 안정적으로
  설명하는가"를 측정하게 됨.
- 난수를 prefix 간 공유(common random numbers)하여 순수 난수 분산이
  아니라 prefix 길이 효과만 차이에 남도록 함.
- `zs[0]=0` 관행은 동일하게 유지(eps[0]=0)해 프로토콜 차이를 최소화.
- 그 외 모든 조건(T=20, 256px, SD v1.4, 학습 60k/평가 96k, 분류기,
  하이퍼파라미터) 동일 — 비교 변인은 노이즈 재주입 여부 하나.

구현: `extract_tre.py --fresh` (서버), 결과는 `features_fresh/` +
`results_fresh.json`으로 저장하여 재현 결과(`results.json`)와 비교.

## 베이스라인 정책

STRE 베이스라인은 직접 재실험하지 않고 **원문(Ren et al.) 공표 수치를
인용**한다 (필요 시 저자가 직접 재실험 예정). 인용 시 프로토콜 차이(데이터
서브셋 구성, 해상도, 학습 규모)가 있을 수 있음을 비교표에 명시할 것.

## 재현 기록 (실험 환경)

- 데이터: GenImage (HF 미러 jzousz/GenImage), 학습 SDv1.4 train 30k+30k
  (seed 42), 평가 8개 생성기 test 12.5k씩 (genimage_test.zip)
- 특징 추출: 원본 O(T²) prefix 복원 의미 보존, 이미지 배치(48)만 병렬화,
  fp32 연산 / fp16 저장
- 분류기: AttentionClassifier(MHSA embed4/head4/2층 + SpatialFocusing +
  ResNet18-4ch), CE loss, Adam 1e-4, batch 16 — LoadDataset.ipynb 레시피,
  head 수만 embed 제약(4의 약수)으로 조정
- 수정한 원본 잠재 버그: (1) SpatialFocusing 5D 입력 불일치 → 채널 평균
  전달, (2) build_dataset.py의 ImageFolder ai/nature 라벨 반전(추출기는
  명시 라벨 사용으로 회피)
