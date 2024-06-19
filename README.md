- # rddcnn_paddle

  A robust deformed CNN for image denoising (CAAI Transactions on Intelligence Technology,2022)

  ## 训练步骤

  ### train 

  ```bash
  python train.py -opt config/train/train_rddcnn.yml
  ```

  ## 测试步骤

  ```bash
  python val.py -opt config/test/test_rddcnn.yml
  ```

  