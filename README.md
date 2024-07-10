# (전자공학회논문지) 트랜스포머 오토인코더와 Diffusion 모델을 사용한 3D CAD 모델 생성 방법 - 실험 코드

정민섭, 김민성, 김지범. (2023). 트랜스포머 오토인코더와 Diffusion 모델을 사용한 3D CAD 모델 생성 방법. 전자공학회논문지, 60(12), 25-28, 10.5573/ieie.2023.60.12.25

오토인코더 학습은 [DeepCAD](https://github.com/ChrisWu1997/DeepCAD)를 참고하세요.

**Diffusion 모델 학습**
```bash
$ python diffusion.py --exp_name EXP-NAME-HERE --ae_ckpt 1000 -g 0
```

**Random generation**
```bash
$ python diffusion.py --exp_name EXP-NAME-HERE --ae_ckpt 1000 --ckpt 200000 --test --n_samples 9000 -g 0

$ python test.py --exp_name EXP-NAME-HERE --mode dec --ckpt 1000 --z_path proj_log/newDeepCAD/lgan_1000/results/fake_z_ckpt200000_num9000.h5 -g 0
```
