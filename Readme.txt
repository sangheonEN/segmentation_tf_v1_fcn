"""

제작자 : 이상헌
연락처 : 010-3234-1906
1. data set을 준비합니다. train, test, valid. 
   -> 저는 image를 불러와 npy형태로 변환하고 train : test : valid = 8 : 1 : 1 비율로 나누었습니다.
2. 각 훈련, test 데이터셋을 patch or raw size 정하고 patch라면 nondeconv_fcn_renual_patch.py, raw size라면 nondeconv_fcn_renual_origin.py 로 훈련과 test를 실시합니다.
3. 각 훈련 code에서 hyperparameters를 argparser로 전달해주어 훈련을 진행합니다.
4. 훈련이 진행 되면 log폴더의 saver에는 weight와 bias가 저장되고 tensorboard에는 train, valid의 loss 값을 그래프화 해주는 tensorboard를 생성합니다. 또한 epoch마다 valid의 예측 image를 pred_mask 폴더에서 볼 수 있습니다.
5. 훈련이 끝난 뒤 가장 validation loss가 낮은 model이 최신으로 저장되고 그 모델을 불러와 다시 test를 실시합니다.
6. test 후에는 accuracy와 pred한 image를 pred_mask에서 볼 수 있습니다.

"""