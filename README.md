# NIA 음성 및 모션 합성 과제 베이스 코드 

음성 데이터로부터 자연스러운 모션을 합성하는 과제의 베이스 코드이며 2017년 LSTM을 활용한 음성 기반 모션 합성 논문([링크](https://www.researchgate.net/publication/320435956_Speech-to-Gesture_Generation_A_Challenge_in_Deep_Learning_Approach_with_Bi-Directional_LSTM))를 바탕으로 구현됨

## 환경 

- Conda 환경 사용을 추천 ([링크](https://www.anaconda.com/products/individual))
- python = 3.7

## 준비 

### 패키지 설치 

```
pip install -r requirements.txt
```

### Pytorch 설치 

공식 pytorch 홈페이지의 [설치 페이지](https://pytorch.org/get-started/previous-versions/#v160)에 들어가서 알맞은 환경의 torch 1.6 버전 설치 

## 작업 폴더 구성 및 설명 

``` 
\utils 
    \collator.py : Dataloader에 필요한 collate 클래스 코드 
    \dataset.py : 전처리 후 데이터에 적용되는 dataset 클래스 코드 
    \feature_helper.py : 음성 및 모션 데이터의 feature를 추출하기 위해 필요한 함수를 담고있는 코드 
    \feature_tools.py : 음성 및 모션 데이터의 feature 특징 추출 파일 
    \Networks.py : 음성 데이터를 입력으로 한 LSTM 기반 모션 합성 네트워크 코드 
    \utils.py : 저장, 데이터 불러오기, 그리기 등 베이스 코드에 필요한 다양한 함수를 담고있는 코드 
\preprocessing.py : wav와 bvh 파일을 numpy 파일로 변경하는 전처리 코드 
\get_reference_npy.py : silence.wav와 hierarchy.bvh 파일을 numpy 파일로 변경하는 전처리 코드 
\train.py : 학습 코드 
\inference.py : 추론 코드 
\requirements.txt : 환경 설정을 위한 패키지 명세 파일
\data_folder : 제공하는 데이터 폴더 
\ref_data_folder : 제공하는 추가 데이터 폴더 
``` 

## 전처리 

1. 제공하는 데이터 폴더(data_folder, ref_data_folder)를 작업 폴더에 저장 

2. 데이터의 구조는 아래와 같아야 함 (한 쌍의 wav 음성 파일과 bvh 모션 파일이 동일한 이름을 갖고 있어야 함)

``` 
\data_folder 
    \train
        \wav 
            \TrainSeq000.wav
            \TrainSeq002.wav
            ...
            \TrainSeq044.wav
        \bvh 
            \TrainSeq000.bvh
            \TrainSeq002.bvh
            ...
            \TrainSeq044.bvh
        \json 
            \TrainSeq000.json
            \TrainSeq002.json
            ...
            \TrainSeq044.json
    \test
        \wav 
            \TestSeq000.wav
            \TestSeq002.wav
            ...
            \TestSeq044.wav
        \bvh 
            \TestSeq000.bvh
            \TestSeq002.bvh
            ...
            \TestSeq044.bvh
        \json 
            \TrainSeq000.json
            \TrainSeq002.json
            ...
            \TrainSeq044.json

\ref_data_folder
    \silence.wav
    \hierarchy.bvh
```

3. 다음을 실행시켜 data_folder에 들어있는 wav와 bvh를 학습에 사용할 동일 이름의 npy 파일로 변경 

```
python preprocessing.py --data_folder data_folder/train --target_folder preprocessed --mfcc_inputs 26 --n_joint 26
```

- data_folder : 학습 과정에서 사용할 audio와 motion 데이터를 포함하는 데이터 폴더 
- target_folder : 전처리 과정으로 생성될 npy 파일이 저장될 폴더 
- mfcc_inputs : wav 파일에서 특징 추출 후 생성되는 mfcc의 channel 수 (default : 26)
- n_joint : 사용하는 모션 데이터의 신체 조인트 수 (default : 26)


4. 다음을 실행하여 무음 silence.wav와 모션의 베이스 포즈인 hierarchy.bvh 파일을 silence.npy와 hierarchy.npy로 변경 

```
python get_reference_npy.py --wav_path ref_data_folder/silence.wav --bvh_path ref_data_folder/hierarchy.bvh --target_folder preprocessed_ref 
```

- wav_path : silence.wav 파일의 경로 
- bvh_path : hierarchy.bvh 파일의 경로 
- target_folder : silence.npy와 hierarchy.npy 파일이 저장될 폴더 

## 학습

```
[Old Model Training]

python train.py --dataset_path preprocessed --results_path results_old --silence_npy_path preprocessed_ref/silence.npy --hierarchy_npy_path preprocessed_ref/hierarchy.npy --context 30 --mfcc_channel 26 --n_joint 26 --epoch 200 --learning_rate 0.01 --weight_decay 0.01 --milestones 300 400
```

```
[New Model Training]

python train.py --dataset_path preprocessed --results_path results_new --silence_npy_path preprocessed_ref/silence.npy --hierarchy_npy_path preprocessed_ref/hierarchy.npy --context 30 --mfcc_channel 26 --n_joint 26 --epoch 200 --learning_rate 0.0001 --milestones 300 400 --weight_decay 0.01 --revised_model
```

- dataset_path : 전처리 과정에서 생성된 npy 폴더 (전처리 3 과정의 결과물)
- results_path : 학습 결과가 저장될 폴더 경로 
- silence_npy_path : 전처리 4 과정 중 silence.wav를 변환해 만든 npy file의 경로 
- hierarchy_npy_path : 전처리 4 과정 중 hierarchy.bvh를 변환해 만든 npy file의 경로
- context : 목적으로하는 시점 전, 후에 추가하는 음성 데이터 양. 한쪽 방향으로의 frame 수를 의미
- 추천 하이퍼 파라메터 
    - epoch : 500 (학습 epoch 수)
    - learning_rate : 0.01 (학습률)
    - weight_decay : 0.01 (weight decay lambda 값 )
    - milestones : 300 400 (0.1배 학습률 변경 epoch 지점)
    - context : 30 (한쪽 방향의 추가 음성 데이터 frame 수)


## 평가 

```
[Old Model Inference]

python inference.py --model_path results_old/train_3/LSTM_Final.ckpt --input_wav data_folder/train/wav/MM_D_E_FF_CC_S335S336_002.wav --hierarchy_bvh_path ref_data_folder/hierarchy.bvh --silence_npy_path preprocessed_ref/silence.npy --mfcc_channel 26 --n_joint 26 --context 30 --output_path results_train_old_002_1203.mp4
```

```
[New Model Inference]

python inference.py --model_path results_new/train_2/LSTM_Final.ckpt --input_wav data_folder/train/wav/MM_D_E_FF_CC_S335S336_002.wav --hierarchy_bvh_path ref_data_folder/hierarchy.bvh --silence_npy_path preprocessed_ref/silence.npy --hidden_size 256 --dropout 0.1 --mfcc_channel 26 --n_joint 26 --context 30 --output_path results_train_002_1202.mp4 --revised_model
```

- model_path : 학습된 모델 경로 
- input_wav : 추론에 사용할 입력 음성
- hierarchy_bvh_path : hierarchy.bvh 파일의 경로 (default : ref_data_folder/hierarchy.bvh)
- silence_npy_path : 전처리 4 과정 중 silence.wav를 변환해 만든 npy file의 경로
- context : 목적으로하는 시점 전, 후에 추가하는 음성 데이터 양. 한쪽 방향으로의 frame 수를 의미 (학습시 사용한 context 값과 같아야 함)
- output_path : 결과 비디오가 저장될 위치 