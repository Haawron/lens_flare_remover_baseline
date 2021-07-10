
## Dataset
- Dataset folder 아래에 다음 파일이 존재
  - sample_submission
  - test_input_img
  - train_input_img
  - train_label_img

##  Code
- model/
  - baseline.py
    - baseline코드 구현
  - common.py
    - 공통적으로 사용하는 residual block등 정의
- utils/
  - loss.py
    - loss 함수 정의 및 반환 => l1,mse 정도만 하면될듯
  - trainer.py
    - trainer 함수 loss,model,argument를 받아서 정의된 에폭만큼 학습 수행
  - dataset.py
    - dataloader 정의
  - utils.py
    - tensorization,augumentation,logger등 
  - psnr.py
    - PSNR을 따로 땜, 아마 나중에 해당 로직을 수정해야 할 수도 있음
- option.py
  - option을 받음
- main.py
  
# 참고
[dacon](https://dacon.io/en/competitions/official/235746/overview/description)<br>
[EDSR](https://github.com/sanghyun-son/EDSR-PyTorch)<br>
[pytorch example](https://github.com/pytorch/examples/tree/master/super_resolution)