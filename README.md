# [[Dacon] 문장 유형 분류AI 경진대회](https://dacon.io/competitions/official/236037/overview/description) - 아최나

---

# 목차
[1. Introduction](https://github.com/wogkr810/Dacon_Sentence_Type_Classification#1-introduction)  
[2. Project Outline](https://github.com/wogkr810/Dacon_Sentence_Type_Classification#2-project-outline)  
[3. Usage & Reproduction](https://github.com/wogkr810/Dacon_Sentence_Type_Classification#3-usage--reproduction)  
[4. Directory Structure](https://github.com/wogkr810/Dacon_Sentence_Type_Classification#4-directory-structure)  
[5. References](https://github.com/wogkr810/Dacon_Sentence_Type_Classification#5-references)  
[6. Retrospect](https://github.com/wogkr810/Dacon_Sentence_Type_Classification#6-retrospect)  

---
# 1. Introduction

## 소개 & 성적

### Score  
![public 4th](https://img.shields.io/badge/PUBLIC-4th-red?style=plastic) ![private 3rd](https://img.shields.io/badge/PRIVATE-3rd-red?style=plastic)

### Public & Private Leaderboard
▶ Public Score  
<img width="889" alt="image" src="https://user-images.githubusercontent.com/46811558/209296974-cd3fba88-ab17-46ff-af12-ba2761058b86.png"> 

▶ Private Score  
<img width="885" alt="image" src="https://user-images.githubusercontent.com/46811558/210361471-3dbcdfa0-0d81-4698-8792-93feab942f59.png">


### Members

이재학|
:-:|
<img src='https://user-images.githubusercontent.com/46811558/197347033-8e0b3742-8450-43b6-aa71-12e3f1745031.jpg' height=100 width=100px></img>|
|[wogkr810](https://github.com/wogkr810)|
|jaehahk810@naver.com|

---
# 2. Project Outline


## 대회 개요


>문장 유형 분류 AI 모델 개발 


## 대회 기간 
> **2022.12.12 ~ 2022.12.23**

## 평가 방법 : Weighted F1 Score
![score](https://user-images.githubusercontent.com/46811558/209297493-be333649-6872-40ef-b5a2-77cf7b9fef33.png)

## 대회 데이터셋
> 뉴스 기사에서 추출한 총 23631(16541+7090)개의 단문 텍스트 데이터  
> 각 데이터에는 *ID*, *문장*, *유형*, *극성*, *시제*, *확실성*, *label* 존재
- train : 16541개
- test : 7090개

> train.csv  
- ID : 샘플 문장 별 고유 ID
- 문장 : 샘플 별 한개의 문장
- 유형 : 문장의 유형 (사실형, 추론형, 대화형, 예측형)
- 극성 : 문장의 극성 (긍정, 부정, 미정)
- 시제 : 문장의 시제 (과거, 현재, 미래)
- 확실성 : 문장의 확실성 (확실, 불확실)
- label : 문장 별 유형, 극성, 시제, 확실성에 대한 Class (총 72개 종류의 Class 존재)
    - Ex) 사실형-긍정-현재-확실

> test.csv
- ID : 샘플 문장 별 고유 ID
- 문장 : 샘플 별 한개의 문장

> sample_submission.csv
- ID : 샘플 문장 별 고유 ID
- label : 예측한 문장 별 유형, 극성, 시제, 확실성에 대한 Class
    - Ex) 사실형-긍정-현재-확실

> 데이터셋 예시
- Train  

![train](https://user-images.githubusercontent.com/46811558/209300364-df10b17f-ea54-4247-add8-20121c4f3556.png)

- Test

![test](https://user-images.githubusercontent.com/46811558/209300367-d32cbe9a-5803-4216-98cc-4ebf5305bf17.png)

---

# 3. Usage & Reproduction


## 사용방법론 및 재현 명령어  
  


### 실험 관련 링크
[Weights & Biases](https://wandb.ai/wogkr810/sen_class?workspace=user-wogkr810)

**모델**  
[klue/roberta-large](http://huggingface.co/klue/roberta-large)

**훈련**  
`python train.py`

**추론(단일 모델)**  
`python inference.py`  

**SOTA 재현 순서 : 5개의 5-Fold 단일모델 -> 하드보팅**  
1. colab Pro Plus -> GPU 등급 : 프리미엄 -> Nvidia A100 GPU    
2. `pip install -r requirements.txt` 명령어 실행  
3. `arguments.py`의 **model_name**을 밑에 있는 항목으로 변경 후 학습(train.py) -> `soft_voting.ipynb` 코드 실행 후 5-Fold 결과물들이 `'./results/soft_ensemble/'` 경로에 생성
    - 첫번째 **model_name** : `'roberta_documnet_mean_max'`
    - 두번째 **model_name** : `'roberta_documnet_weighted'`
    - 세번째 **model_name** : `'roberta_documnet_concat_hidden'`
    - 네번째 **model_name** : `'roberta_documnet_sds'`
    - 두번째 **model_name** : `'roberta_documnet_linear'`  
4. 3번의 과정 이후에 csv파일들이 저장됐다면, `hard_voting_ipynb` 코드 실행 후 최종 하드보팅 결과물 `'./results/hard_ensemble/'` 경로에 생성
5. **WandB를 사용하려면, wandb 관련 주석 해제 후 login key값, name, project 설정 후 실행**

**최종 Score**
- 5-Fold
    - `'roberta_documnet_mean_max'` 
        - Public : 0.7555
        - Private : 0.7550
    - `'roberta_documnet_mean_weighted'` 
        - Public : 0.7509
        - Private : 0.7524
    - `'roberta_documnet_concat_hidden'` 
        - Public : 0.7475652747
        - Private : 0.7537354181
    - `'roberta_documnet_mean_sds'` 
        - Public : 0.7475
        - Private : 0.7537
    - `'roberta_documnet_linear'` 
        - Public : 0.7451894032
        - Private : 0.7498835687
- Hard_Voting(5개의 5-Fold)
    - Public : 0.758202060457
    - Private : 0.7574623641

## 하드웨어 & 라이브러리

**Colab Pro Plus**
- CPU : 6C
- GPU : A100-SXM4-40GB(1C)
- 용량 : 100GB(구글 드라이브 One Basic)
- OS : Linux-5.10.133+-x86_64-with-glibc2.27
- Python : 3.8.16
- W&B CLI Version : 0.13.7


---

# 4. Directory Structure

## 디렉토리 구조 & 파일 설명

```
USER/
├── trainer.py
├── train.py
├── inference.py
├── model.py
├── custom_dataset.py
├── arguments.py
├── [Baseline]_TfidfVectorizer + MLP.ipynb
├── soft_voting.ipynb
├── hard_voting.ipynb
├── EDA.ipynb
├── README.md
├── requirements.txt
├── .gitignore
│
├── assets
│   └── 아최나_발표자료.pdf
│
├── utils
│   ├── heads.py
│   ├── loss.py
│   ├── pooling.py
│   └── scheduler.py
│
├── results
│   ├── hard_ensemble
│   │   └──(sample)hard_results_12_22_23_39.csv
│   ├── single_model
│   │   └──(sample)submission12_23_16_14.csv
│   └── soft_ensemble
│       ├──(sample)roberta_document_mean_max_12_22_23_27.csv
│       ├──(sample)roberta_document_weighted_12_22_23_27.csv
│       ├──(sample)roberta_document_concat_hidden_12_22_23_27.csv
│       ├──(sample)roberta_document_sds_12_22_23_27.csv
│       └──(sample)roberta_document_linear_12_22_16_59.csv
│
├── saved
│   ├── roberta_document_concat_hidden
│   │   └──k(0,1,2,3,4)_roberta_document_concat_hidden
│   │       ├──model_(0,1,2,3,4)_mean_f1.pth
│   │       └──model(best_scores)_(0,1,2,3,4)_mean_f1.pth
│   ├── roberta_document_concat_hidden
│   │   └──k(0,1,2,3,4)_roberta_document_concat_hidden
│   │       ├──model_(0,1,2,3,4)_mean_f1.pth
│   │       └──model(best_scores)_(0,1,2,3,4)_mean_f1.pth
│   ├── roberta_document_concat_hidden
│   │   └──k(0,1,2,3,4)_roberta_document_concat_hidden
│   │       ├──model_(0,1,2,3,4)_mean_f1.pth
│   │       └──model(best_scores)_(0,1,2,3,4)_mean_f1.pth
│   ├── roberta_document_concat_hidden
│   │   └──k(0,1,2,3,4)_roberta_document_concat_hidden
│   │       ├──model_(0,1,2,3,4)_mean_f1.pth
│   │       └──model(best_scores)_(0,1,2,3,4)_mean_f1.pth
│   └── roberta_document_concat_hidden
│       └──k(0,1,2,3,4)_roberta_document_concat_hidden
│           ├──model_(0,1,2,3,4)_mean_f1.pth
│           └──model(best_scores)_(0,1,2,3,4)_mean_f1.pth
│
├── data
│   ├── (sample)test.csv
│   ├── (sample)train.csv
│   └──sample_submission.csv
│ 
└──────────────────────────
```
- `trainer.py`

  - `train.py`에서 사용되며, torch기반으로 training 및 validation loop을 실행합니다.

- `train.py`

  - 모델 학습을 실행하는 코드입니다.
  - Epoch 마다 validation이 실행되며, 가중치 파일은 f'./saved/args.model_name/k{fold_index}_{args.model_name}' 폴더에 저장됩니다.

- `inference.py`

  - 5-Fold를 적용하지 않은 단일모델의 추론을 실행하는 코드입니다.
  - 학습된 model 가중치를 통해 prediction하고, 예측한 결과를 csv파일로 저장하는 코드입니다.
  - 최종 csv파일은 './results/single_model' 폴더에 생성됩니다.

- `model.py`

  - BaseModel : 베이스라인 기반 모델입니다.
  - Robertainear : transformers.models.roberta.modeling_roberta의 RobertaModel 및 RobertaPreTrainedModel 클래스를 가져와, 커스텀 레이어를 적용한 모델입니다.
    - `arguments` : 
        - roberta_class
        - roberta_dacon
        - roberta_linear
        - roberta_sds
  - RobertaDocument : transformers.models.roberta.modeling_roberta의 RobertaModel 및 RobertaPreTrainedModel 클래스를 가져와, 커스텀 레이어 적용 및 hidden states & pooling layer를 적용한 모델입니다.
    - `arguments` : 
        - roberta_document_linear
        - roberta_document_sds
        - roberta_document_concat_hidden
        - roberta_document_mean_max
        - roberta_document_lstm
        - roberta_document_weighted

- `custom_dataset.py`

  - make_roberta_data(args) : args를 받아, EDA를 통한 중복 문장 제거 이후, Label encodr를 적용하여 반환하는 함수입니다.
  - RobertaDataset : pytorch의 Dataset 클래스를 상속받아, 모드(train,test)에 따라 label 반환 및 토크나이즈한 문장의 `input_ids`, `attention_mask`를 반환합니다.

- `arguments.py`

  - 필요한 arguments들을 정의한 파일입니다.
  - 항목이 많아, [하단에 표](https://github.com/wogkr810/Dacon_Sentence_Type_Classification#arguments)로 기재하였습니다. 


- `[Baseline]_TfidfVectorizer + MLP.ipynb`

  - 베이스라인 코드입니다.

- `soft_voting.ipynb`

  - 저장된 weights들의 확률을 이용하여 soft_voting을 적용하는 파일입니다.
  - K-fold를 적용하기 위해 제작하였습니다.

- `hard_voting.ipynb`

  - 추론 후 나온 csv파일들을 이용하여 최다빈도 보팅기법을 적용하는 파일입니다.

- `EDA.ipynb`

  - 탐색적 데이터 분석을 적용한 파일입니다.
  - 데이터 길이(max_input_length), 형태, 길이분포, 통계치, 결측치, 중복, 라벨분포 등을 시각화하여 적용했으며, 데이터를 살펴보며 전처리에 관련된 인사이트를 얻었습니다.
  - 적용한 파일을 [코드공유 게시판](https://dacon.io/competitions/official/236037/codeshare/7317?page=1&dtype=recent)에 업로드하였습니다.

- `requirements.txt`

  - 사용할 라이브러리들이 기재된 파일입니다.
  - `pip install -r requirements.txt` 명령어를 통해 실행합니다.

- `assets/`

  - 발표자료가 저장된 디렉토리 입니다.


- `utils/`

    - 커스텀 loss&scheduler 및 커스텀 모델을 위한 heads&pooling layer 들이 있는 디렉토리입니다.
	- `heads.py`  
	    - 모델에 사용된 layer 및 head들을 customize할 수 있는 class들을 정의한 파일입니다.
        - Hidden states 및 layer 등을 custom하게 사용합니다.
	- `pooling.py` 
	    - 모델의 pooling layer를 customize할 수 있는 class들을 정의한 파일입니다.
        - Mean-max, LSTM, Weighted layer pooling을 custom하게 사용합니다.
	- `loss.py`
        - 모델의 criterion을 변경할 수 있는 class들을 정의한 파일입니다.
        - crossentropy, focal , f1, label_smoothing loss을 custom하게 사용합니다.
	- `scheduler.py`
        - 모델의 scheduler을 변경할 수 있는 class들을 정의한 파일입니다.
        - huggingface 기반 코드와 다르게, pytorch 기반으로 작성했기에 step에 따른 학습이 진행되지 않아 사용하지 않았습니다.

- `results/`

    - 추론 후 나온 제출 csv 파일들이 있는 디렉토리입니다.
	- `hard_ensemble/`  
	    - `hard_voting.ipynb` 을 실행하여 나온 제출물이 있는 디렉토리입니다.
	- `single_model/`
        - `inference.py` 을 실행하여 나온 단일모델 제출물이 있는 디렉토리입니다. 
	- `soft_ensemble/`
	    - `soft_voting.ipynb` 을 실행하여 나온 제출물이 있는 디렉토리입니다.

- `saved/`

    - 학습 도중, validation loop에서 산출된 weights들이 있는 디렉토리입니다.
    - 사용한 모델의 이름(args.model_name)으로 디렉토리가 생성되며, K-fold를 적용할경우, fold_index에 따라 세부 디렉토리가 적용됩니다.
        - 각 디렉토리에는 '유형', '극성', '시제', '확실성' 각각의 weighted f1 score을 평균낸 점수를 각 epoch 마다 비교하여, 최대치를 달성한 epoch에서 model 정보 및 model의 state_dict 정보를 저장합니다.

- `data/`

    - 모델에 사용된 데이터들이 저장된 디렉토리입니다.
    - `sample_submission.csv` 파일을 적용하여 최종 추론 파일을 생성할 수 있습니다.



## Arguments

### training_args : 학습에 필요한 arguments들 입니다.

|          argument           | description                                               |
| :-------------------------: | :-------------------------------------------------------- |
|          seed               | 랜덤시드                        |
|          batch_size         | 배치사이즈                          |
|          learning_rate      | 러닝레이트                               |
|          epochs             | iteration 수            |
|          split_ratio        | train,validation split ratio, StratifiedKFold를 사용하므로 사용 x |
|          scheduler_type     | optimizer에 적용된 scheduler                |
|          optimizer_type     | optimizer 종류         |
|          warmup_steps       | learning rate warm up, ReduceLROnPlateau를 사용하므로 사용 x     |
|          use_kfold          | k-fold 적용 여부    |
|          print_name         | wandb logging name을 커스텀하기 위한 string   |


### Path_args : 경로 arguments들 입니다.

|          argument           | description                                               |
| :-------------------------: | :-------------------------------------------------------- |
|          data_path          | 데이터 경로                        |
|          output_path        | 추론 후 결과물 경로                          |
|          saved_path         | validation 이후 weights 저장 경로                              |


### Model_args : 모델 관련 arguments들 입니다.

|          argument           | description                                               |
| :-------------------------: | :-------------------------------------------------------- |
|          max_input_length   | 토크나이저에 적용될 최대 토큰 길이입니다. EDA이후 256로 적용했지만, gpu 환경에 맞게 128로 수정했습니다.                        |
|          PLM                | pretrained model 경로                          |
|          loss_name          | 적용할 loss 종류(crossentropy, focal, f1, label_smoothing)            |
|          use_tfidf          | BaseModel을 사용할 경우 True, RobertaModel을 사용할 경우 False            |
|          use_roberta        | Roberta관련 모델을 사용할 경우 True          |
|          model_name         | 사용할 모델 종류                              |


---

# 5. References

## Reference

- [Smart Loss](https://github.com/archinetai/smart-pytorch)
- [Balanced Loss](https://github.com/Roche/BalancedLossNLP)


---

# 6. Retrospect


## 회고
- 아쉬운 점
    - 라벨 불균형이 심한데, 언더샘플링 & 오버샘플링 해보지 않은 것.
        - 라벨 불균형이 심한 것 따로 학습하여 에폭을 적게하거나, tense를 제외하고 라벨의 극심한 불균형이 있는 열들을 다른모델을 쓰거나 기준을 다르게 잡았어도 좋았을 듯.
        - 언더샘플링을 적용하기에, 4개의 열에 의존된 라벨&극심한 불균형으로 기준을 잡지못함.
        - 데이터 분석 결과, 문장은 기사에서 추출한 정제된 문장. 따라서, 부족한 label은 eda, aeda 정도는 적용했어도 좋았을 듯.
    - scheduler 적용시에, huggingface vs pytorch 모델링에서, step 부분 로깅 그냥 넘긴 것.
    - 라벨 64개로 분류해보지 않은 것.
        - 전체 조합은 72개이지만, 실제로 train.csv에는 64개의 label이 존재함.
        - 혹시 8개가 나올까 하는 생각 & 각각의 모델이 성능이 좋을 것이란 생각에 그냥 진행 
    - paperswithcode SOTA(Balanced Loss with Long Tailed Dataset) 적용 못한 것.
    - 다양한 loss 적용 못한 것.
        - 불균형에 따라 다른 loss를 적용하면 scale이 다르지 않을 까 하는 생각에 적용X.
        - 하지만, 4개의 loss가 같더라도, focal,f1,label_smoothing등은 적용하여 제출했어야 함.
        - [관련 블로그](https://supermemi.tistory.com/184)를 참고하면, `nn.Crossentropy` 문서에는, 코드기준 logit과 label shape이 같아야하지만, `F.cross_entropy` 는 달라도 가능함. 하지만, `F.binary_cross_entropy`는 shape이 같아야함. 따라서, 이진분류에 효과적이라는 BCEloss는 적용하지 않음.
    - 하이퍼 파라미터 튜닝 못한 것.
        - 한정된 시간&GPU 자원으로 하지 못함.
        - 학습 시간을 줄이기 위해, 배치 64 및 max 128로 진행했지만, 불균형 데이터로 인해 오히려 큰 배치가 효과를 발휘 했을수도 있음.
- 느낀 점
    - 확실히 대회가 공부가 많이 된다.
    - roberta custom, smart loss 등 까고 보니 별로 안어렵다. smart는 심지어 `ForSequenceClassification` 으로 적용가능
- 얻은 것
    - 부스트캠프 시작하고 AI를 공부하면서 많은 대회에서 모델을 거의 건드려보지 못했는데, 이번 대회에서 직접 로버타 모델 하나하나 입력&출력 보면서 커스텀한 것
- 다음에 할 것
    - pytorch여도 step으로 scheduler & optimizer 적용, step기준 evaluation 적용해보기
    - 데이터 전처리 및 증강
- 아직도 모르겠는 점
    - `model.py` 에서 실행에 관계없는 class의 코드 건드릴 때마다 왜 재현이 안됐는지. 바뀔때마다 재현 확인하느라 시간&자원 낭비함.
    - learning_rate, batch_size, max_input_length에 대한 명확한 기준. GPU별 차이
    - `klue/roberta-large` 모델의 에폭. KERC에서 6으로했지만, 다른 분들은 대부분 3정도. 이번대회는 문장의 길이가 더 짧았으므로 더 적게 적용하려 했지만, 에폭7했을 때 잘나온 기억때문에 방황함.
    - 채점에 대한 기준. weighted f1 score인건 알겠지만, 4개의 열에 대해 평균을 내는건가.
    - multiclass vs multilabel에 대한 기준. 우리의 Task는 Multiclass with Multilabel인 것 같음.
        - paperswithcode에서 text-classification vs multi-label text-classification 고민.
