Model Evaluation & Training Logs

Script: ablation_narrowband

PS E:\Res> python ablation_narrowband.py
Using device: cuda

============================================================
Config: plain  (augmentation=False, batchnorm=False)
Checkpoint selection band: -12 to -6 dB
============================================================
  Epoch   1: Val Acc 40.79%, Band(-12..-6dB) Loss 2.1863 (Acc 16.80%) (best-band)
  Epoch   2: Val Acc 45.62%, Band(-12..-6dB) Loss 2.0668 (Acc 20.97%) (best-band)
  Epoch   3: Val Acc 47.24%, Band(-12..-6dB) Loss 2.0067 (Acc 22.90%) (best-band)
  Epoch   4: Val Acc 48.76%, Band(-12..-6dB) Loss 1.9510 (Acc 23.95%) (best-band)
  Epoch   5: Val Acc 50.96%, Band(-12..-6dB) Loss 1.9056 (Acc 26.71%) (best-band)
  Epoch   6: Val Acc 50.38%, Band(-12..-6dB) Loss 1.9373 (Acc 25.68%)
  Epoch   7: Val Acc 53.58%, Band(-12..-6dB) Loss 1.8997 (Acc 27.57%) (best-band)
  Epoch   8: Val Acc 53.89%, Band(-12..-6dB) Loss 1.8492 (Acc 30.21%) (best-band)
  Epoch   9: Val Acc 54.31%, Band(-12..-6dB) Loss 1.8456 (Acc 27.64%) (best-band)
  Epoch  10: Val Acc 55.12%, Band(-12..-6dB) Loss 1.8542 (Acc 29.24%)
  Epoch  11: Val Acc 55.63%, Band(-12..-6dB) Loss 1.7963 (Acc 31.21%) (best-band)
  Epoch  12: Val Acc 55.64%, Band(-12..-6dB) Loss 1.8237 (Acc 29.58%)
  Epoch  13: Val Acc 56.27%, Band(-12..-6dB) Loss 1.7982 (Acc 31.66%)
  Epoch  14: Val Acc 56.13%, Band(-12..-6dB) Loss 1.7987 (Acc 31.36%)
  Epoch  15: Val Acc 55.89%, Band(-12..-6dB) Loss 1.7901 (Acc 31.88%) (best-band)
  Epoch  16: Val Acc 56.38%, Band(-12..-6dB) Loss 1.7980 (Acc 30.98%)
  Epoch  17: Val Acc 56.61%, Band(-12..-6dB) Loss 1.7868 (Acc 32.30%) (best-band)
  Epoch  18: Val Acc 56.53%, Band(-12..-6dB) Loss 1.7950 (Acc 32.50%)
  Epoch  19: Val Acc 56.46%, Band(-12..-6dB) Loss 1.7810 (Acc 31.81%) (best-band)
  Epoch  20: Val Acc 56.45%, Band(-12..-6dB) Loss 1.7750 (Acc 30.89%) (best-band)
  Epoch  21: Val Acc 56.19%, Band(-12..-6dB) Loss 1.8135 (Acc 30.40%)
  Epoch  22: Val Acc 56.15%, Band(-12..-6dB) Loss 1.7896 (Acc 31.26%)
  Epoch  23: Val Acc 56.65%, Band(-12..-6dB) Loss 1.7781 (Acc 31.96%)
  Epoch  24: Val Acc 56.58%, Band(-12..-6dB) Loss 1.7880 (Acc 31.70%)
  Epoch  25: Val Acc 56.34%, Band(-12..-6dB) Loss 1.7829 (Acc 31.75%)
  Epoch  26: Val Acc 56.82%, Band(-12..-6dB) Loss 1.7635 (Acc 32.78%) (best-band)
  Epoch  27: Val Acc 56.78%, Band(-12..-6dB) Loss 1.7668 (Acc 31.71%)
  Epoch  28: Val Acc 56.87%, Band(-12..-6dB) Loss 1.7632 (Acc 31.98%) (best-band)
  Epoch  29: Val Acc 57.02%, Band(-12..-6dB) Loss 1.7543 (Acc 33.03%) (best-band)
  Epoch  30: Val Acc 56.94%, Band(-12..-6dB) Loss 1.7850 (Acc 31.79%)
  Epoch  31: Val Acc 57.08%, Band(-12..-6dB) Loss 1.7851 (Acc 31.43%)
  Epoch  32: Val Acc 56.29%, Band(-12..-6dB) Loss 1.7944 (Acc 30.80%)
  Epoch  33: Val Acc 56.78%, Band(-12..-6dB) Loss 1.7651 (Acc 31.53%)
  Epoch  34: Val Acc 56.84%, Band(-12..-6dB) Loss 1.7660 (Acc 31.66%)
  Epoch  35: Val Acc 57.12%, Band(-12..-6dB) Loss 1.7668 (Acc 31.90%)
  Epoch  36: Val Acc 57.11%, Band(-12..-6dB) Loss 1.7624 (Acc 31.60%)
  Epoch  37: Val Acc 57.23%, Band(-12..-6dB) Loss 1.7609 (Acc 31.92%)
  Epoch  38: Val Acc 57.11%, Band(-12..-6dB) Loss 1.7685 (Acc 31.71%)
  Epoch  39: Val Acc 57.14%, Band(-12..-6dB) Loss 1.7570 (Acc 32.91%)

  Epoch  40: Val Acc 56.63%, Band(-12..-6dB) Loss 1.7779 (Acc 31.19%)
  Epoch  41: Val Acc 57.44%, Band(-12..-6dB) Loss 1.7703 (Acc 32.39%)
  Epoch  42: Val Acc 57.16%, Band(-12..-6dB) Loss 1.7609 (Acc 32.16%)
  Epoch  43: Val Acc 56.56%, Band(-12..-6dB) Loss 1.7979 (Acc 29.91%)
  Epoch  44: Val Acc 56.65%, Band(-12..-6dB) Loss 1.7736 (Acc 31.43%)
  Epoch  45: Val Acc 56.79%, Band(-12..-6dB) Loss 1.7899 (Acc 30.96%)
  Epoch  46: Val Acc 57.12%, Band(-12..-6dB) Loss 1.7657 (Acc 31.34%)
  Epoch  47: Val Acc 56.84%, Band(-12..-6dB) Loss 1.7625 (Acc 31.98%)
  Epoch  48: Val Acc 57.47%, Band(-12..-6dB) Loss 1.7787 (Acc 31.71%)
  Epoch  49: Val Acc 57.25%, Band(-12..-6dB) Loss 1.7852 (Acc 30.42%)
  Stopped: no band improvement in 20 epochs (best was epoch 29).
  >> Final test accuracy: 57.49%

============================================================
Config: aug_only  (augmentation=True, batchnorm=False)
Checkpoint selection band: -12 to -6 dB
============================================================
  Epoch   1: Val Acc 48.07%, Band(-12..-6dB) Loss 2.0231 (Acc 22.54%) (best-band)
  Epoch   2: Val Acc 52.28%, Band(-12..-6dB) Loss 1.9312 (Acc 26.50%) (best-band)
  Epoch   3: Val Acc 53.46%, Band(-12..-6dB) Loss 1.9153 (Acc 24.49%) (best-band)
  Epoch   4: Val Acc 55.18%, Band(-12..-6dB) Loss 1.8478 (Acc 29.58%) (best-band)
  Epoch   5: Val Acc 55.75%, Band(-12..-6dB) Loss 1.8700 (Acc 27.61%)
  Epoch   6: Val Acc 56.19%, Band(-12..-6dB) Loss 1.8617 (Acc 28.17%)
  Epoch   7: Val Acc 56.55%, Band(-12..-6dB) Loss 1.8833 (Acc 27.40%)
  Epoch   8: Val Acc 56.47%, Band(-12..-6dB) Loss 1.8679 (Acc 28.21%)
  Epoch   9: Val Acc 56.20%, Band(-12..-6dB) Loss 1.8227 (Acc 29.73%) (best-band)
  Epoch  10: Val Acc 56.43%, Band(-12..-6dB) Loss 1.8565 (Acc 27.66%)
  Epoch  11: Val Acc 56.54%, Band(-12..-6dB) Loss 1.8378 (Acc 29.16%)
  Epoch  12: Val Acc 56.83%, Band(-12..-6dB) Loss 1.8597 (Acc 28.53%)
  Epoch  13: Val Acc 55.94%, Band(-12..-6dB) Loss 1.8587 (Acc 28.28%)
  Epoch  14: Val Acc 56.25%, Band(-12..-6dB) Loss 1.9168 (Acc 26.48%)
  Epoch  15: Val Acc 56.94%, Band(-12..-6dB) Loss 1.8361 (Acc 29.31%)
  Epoch  16: Val Acc 56.80%, Band(-12..-6dB) Loss 1.8680 (Acc 28.81%)
  Epoch  17: Val Acc 56.91%, Band(-12..-6dB) Loss 1.8658 (Acc 28.66%)
  Epoch  18: Val Acc 56.69%, Band(-12..-6dB) Loss 1.8607 (Acc 27.81%)
  Epoch  19: Val Acc 57.20%, Band(-12..-6dB) Loss 1.8813 (Acc 27.87%)
  Epoch  20: Val Acc 57.48%, Band(-12..-6dB) Loss 1.8569 (Acc 29.65%)
  Epoch  21: Val Acc 57.00%, Band(-12..-6dB) Loss 1.8537 (Acc 29.24%)
  Epoch  22: Val Acc 57.31%, Band(-12..-6dB) Loss 1.8866 (Acc 28.02%)
  Epoch  23: Val Acc 57.95%, Band(-12..-6dB) Loss 1.9081 (Acc 27.81%)
  Epoch  24: Val Acc 58.08%, Band(-12..-6dB) Loss 1.8973 (Acc 28.19%)
  Epoch  25: Val Acc 58.13%, Band(-12..-6dB) Loss 1.8927 (Acc 28.71%)
  Epoch  26: Val Acc 58.30%, Band(-12..-6dB) Loss 1.9171 (Acc 28.00%)
  Epoch  27: Val Acc 58.05%, Band(-12..-6dB) Loss 1.9270 (Acc 28.11%)
  Epoch  28: Val Acc 58.19%, Band(-12..-6dB) Loss 1.9040 (Acc 28.26%)
  Epoch  29: Val Acc 57.74%, Band(-12..-6dB) Loss 1.9423 (Acc 27.49%)
  Stopped: no band improvement in 20 epochs (best was epoch 9).
  >> Final test accuracy: 56.77%

============================================================
Config: bn_only  (augmentation=False, batchnorm=True)
Checkpoint selection band: -12 to -6 dB
============================================================
  Epoch   1: Val Acc 53.37%, Band(-12..-6dB) Loss 1.8475 (Acc 30.78%) (best-band)
  Epoch   2: Val Acc 55.39%, Band(-12..-6dB) Loss 1.8288 (Acc 28.21%) (best-band)
  Epoch   3: Val Acc 56.10%, Band(-12..-6dB) Loss 1.8162 (Acc 28.88%) (best-band)

  Epoch   4: Val Acc 56.44%, Band(-12..-6dB) Loss 1.8019 (Acc 30.29%) (best-band)
  Epoch   5: Val Acc 57.88%, Band(-12..-6dB) Loss 1.7912 (Acc 30.08%) (best-band)
  Epoch   6: Val Acc 57.67%, Band(-12..-6dB) Loss 1.7884 (Acc 30.06%) (best-band)
  Epoch   7: Val Acc 58.48%, Band(-12..-6dB) Loss 1.7709 (Acc 31.55%) (best-band)
  Epoch   8: Val Acc 59.27%, Band(-12..-6dB) Loss 1.7661 (Acc 31.51%) (best-band)
  Epoch   9: Val Acc 60.31%, Band(-12..-6dB) Loss 1.7693 (Acc 31.53%)
  Epoch  10: Val Acc 60.26%, Band(-12..-6dB) Loss 1.7787 (Acc 30.70%)
  Epoch  11: Val Acc 60.06%, Band(-12..-6dB) Loss 1.7669 (Acc 31.30%)
  Epoch  12: Val Acc 59.63%, Band(-12..-6dB) Loss 1.7711 (Acc 31.58%)
  Epoch  13: Val Acc 60.33%, Band(-12..-6dB) Loss 1.7621 (Acc 31.79%) (best-band)
  Epoch  14: Val Acc 60.52%, Band(-12..-6dB) Loss 1.7769 (Acc 32.16%)
  Epoch  15: Val Acc 60.38%, Band(-12..-6dB) Loss 1.7953 (Acc 31.40%)
  Epoch  16: Val Acc 60.73%, Band(-12..-6dB) Loss 1.7749 (Acc 30.93%)
  Epoch  17: Val Acc 61.25%, Band(-12..-6dB) Loss 1.7597 (Acc 32.20%) (best-band)
  Epoch  18: Val Acc 60.91%, Band(-12..-6dB) Loss 1.7750 (Acc 31.58%)
  Epoch  19: Val Acc 60.95%, Band(-12..-6dB) Loss 1.7663 (Acc 31.62%)
  Epoch  20: Val Acc 61.37%, Band(-12..-6dB) Loss 1.7519 (Acc 32.56%) (best-band)
  Epoch  21: Val Acc 61.17%, Band(-12..-6dB) Loss 1.7696 (Acc 31.85%)
  Epoch  22: Val Acc 60.86%, Band(-12..-6dB) Loss 1.7686 (Acc 31.47%)
  Epoch  23: Val Acc 60.71%, Band(-12..-6dB) Loss 1.7765 (Acc 32.33%)
  Epoch  24: Val Acc 61.16%, Band(-12..-6dB) Loss 1.8036 (Acc 31.11%)
  Epoch  25: Val Acc 60.94%, Band(-12..-6dB) Loss 1.7699 (Acc 32.16%)
  Epoch  26: Val Acc 60.95%, Band(-12..-6dB) Loss 1.8210 (Acc 29.58%)
  Epoch  27: Val Acc 61.24%, Band(-12..-6dB) Loss 1.7630 (Acc 31.98%)
  Epoch  28: Val Acc 61.23%, Band(-12..-6dB) Loss 1.7914 (Acc 31.85%)
  Epoch  29: Val Acc 61.09%, Band(-12..-6dB) Loss 1.7839 (Acc 31.25%)
  Epoch  30: Val Acc 61.42%, Band(-12..-6dB) Loss 1.7526 (Acc 32.67%)
  Epoch  31: Val Acc 61.14%, Band(-12..-6dB) Loss 1.7921 (Acc 31.10%)
  Epoch  32: Val Acc 61.41%, Band(-12..-6dB) Loss 1.7672 (Acc 31.70%)
  Epoch  33: Val Acc 61.43%, Band(-12..-6dB) Loss 1.7779 (Acc 31.34%)
  Epoch  34: Val Acc 60.75%, Band(-12..-6dB) Loss 1.7967 (Acc 30.80%)
  Epoch  35: Val Acc 61.28%, Band(-12..-6dB) Loss 1.7727 (Acc 32.09%)
  Epoch  36: Val Acc 61.24%, Band(-12..-6dB) Loss 1.8122 (Acc 31.04%)
  Epoch  37: Val Acc 60.98%, Band(-12..-6dB) Loss 1.8005 (Acc 31.53%)
  Epoch  38: Val Acc 61.36%, Band(-12..-6dB) Loss 1.8230 (Acc 31.02%)
  Epoch  39: Val Acc 60.91%, Band(-12..-6dB) Loss 1.7954 (Acc 31.40%)
  Epoch  40: Val Acc 61.28%, Band(-12..-6dB) Loss 1.8187 (Acc 30.70%)
  Stopped: no band improvement in 20 epochs (best was epoch 20).
  >> Final test accuracy: 62.12%

============================================================
Config: aug_and_bn  (augmentation=True, batchnorm=True)
Checkpoint selection band: -12 to -6 dB
============================================================
  Epoch   1: Val Acc 55.74%, Band(-12..-6dB) Loss 1.8748 (Acc 28.17%) (best-band)
  Epoch   2: Val Acc 56.96%, Band(-12..-6dB) Loss 1.8381 (Acc 29.74%) (best-band)
  Epoch   3: Val Acc 59.45%, Band(-12..-6dB) Loss 1.8497 (Acc 29.05%)
  Epoch   4: Val Acc 60.73%, Band(-12..-6dB) Loss 1.8288 (Acc 29.74%) (best-band)
  Epoch   5: Val Acc 60.93%, Band(-12..-6dB) Loss 1.7891 (Acc 31.02%) (best-band)
  Epoch   6: Val Acc 61.15%, Band(-12..-6dB) Loss 1.8011 (Acc 32.05%)
  Epoch   7: Val Acc 61.60%, Band(-12..-6dB) Loss 1.7832 (Acc 31.19%) (best-band)
  Epoch   8: Val Acc 61.31%, Band(-12..-6dB) Loss 1.8179 (Acc 29.82%)
  Epoch   9: Val Acc 61.26%, Band(-12..-6dB) Loss 1.8228 (Acc 30.59%)
  Epoch  10: Val Acc 61.21%, Band(-12..-6dB) Loss 1.8631 (Acc 28.92%)
  Epoch  11: Val Acc 61.69%, Band(-12..-6dB) Loss 1.8388 (Acc 30.40%)
  Epoch  12: Val Acc 61.27%, Band(-12..-6dB) Loss 1.8072 (Acc 31.08%)

  Epoch  13: Val Acc 61.47%, Band(-12..-6dB) Loss 1.8830 (Acc 28.99%)
  Epoch  14: Val Acc 61.40%, Band(-12..-6dB) Loss 1.8559 (Acc 29.71%)
  Epoch  15: Val Acc 61.08%, Band(-12..-6dB) Loss 1.8643 (Acc 30.08%)
  Epoch  16: Val Acc 61.16%, Band(-12..-6dB) Loss 1.8685 (Acc 30.42%)
  Epoch  17: Val Acc 61.35%, Band(-12..-6dB) Loss 1.8815 (Acc 29.95%)
  Epoch  18: Val Acc 60.77%, Band(-12..-6dB) Loss 1.8957 (Acc 30.16%)
  Epoch  19: Val Acc 60.94%, Band(-12..-6dB) Loss 1.9293 (Acc 29.76%)
  Epoch  20: Val Acc 61.00%, Band(-12..-6dB) Loss 1.9092 (Acc 29.67%)
  Epoch  21: Val Acc 60.17%, Band(-12..-6dB) Loss 1.9527 (Acc 29.16%)
  Epoch  22: Val Acc 60.86%, Band(-12..-6dB) Loss 1.9848 (Acc 28.60%)
  Epoch  23: Val Acc 60.48%, Band(-12..-6dB) Loss 1.9761 (Acc 28.64%)
  Epoch  24: Val Acc 60.91%, Band(-12..-6dB) Loss 1.9878 (Acc 29.29%)
  Epoch  25: Val Acc 60.77%, Band(-12..-6dB) Loss 1.9563 (Acc 29.28%)
  Epoch  26: Val Acc 60.89%, Band(-12..-6dB) Loss 1.9447 (Acc 29.86%)
  Epoch  27: Val Acc 60.78%, Band(-12..-6dB) Loss 1.9747 (Acc 29.95%)
  Stopped: no band improvement in 20 epochs (best was epoch 7).
  >> Final test accuracy: 62.31%

============================================================
SUMMARY (narrow-band checkpoint selection: -12 to -6 dB)
============================================================
Config            Test Acc
plain               57.49%
aug_only            56.77%
bn_only             62.12%
aug_and_bn          62.31%

Config             -20    -18    -16    -14    -12    -10     -8     -6     -4
-2      0      2      4      6      8     10     12     14     16     18
plain            10.8%  10.1%  10.4%  13.4%  17.0%  26.2%  38.5%  50.2%  61.3%  70.7%
79.1%  83.4%  83.0%  83.9%  84.3%  85.4%  85.3%  84.8%  84.7%  85.6%
aug_only         10.8%  10.0%   9.4%  12.2%  15.7%  25.3%  34.4%  45.2%  61.5%  72.3%
76.2%  82.1%  83.6%  83.0%  85.7%  85.1%  85.9%  84.4%  85.9%  84.6%
bn_only          11.0%  10.5%  10.7%  13.3%  16.4%  24.8%  37.6%  54.1%  68.3%  79.1%
87.7%  91.1%  91.2%  91.8%  91.6%  92.6%  92.3%  92.0%  92.2%  92.1%
aug_and_bn       10.4%   9.5%  10.0%  12.9%  16.3%  23.1%  36.5%  50.6%  69.7%  79.5%
87.9%  91.7%  92.5%  92.5%  92.8%  93.9%  93.6%  93.3%  93.8%  93.4%

Script: ablation_reduced

PS E:\Res> python ablation_reduced.py
Using device: cuda

============================================================
Config: plain  (augmentation=False, batchnorm=False)
Checkpoint selection band: -12 to -6 dB
============================================================
  Trainable parameters: 21,123
  Epoch   1: Val Acc 38.95%, Band(-12..-6dB) Loss 2.1491 (Acc 17.31%) (best-band)
  Epoch   2: Val Acc 44.50%, Band(-12..-6dB) Loss 2.0732 (Acc 19.86%) (best-band)
  Epoch   3: Val Acc 46.05%, Band(-12..-6dB) Loss 2.0566 (Acc 19.34%) (best-band)
  Epoch   4: Val Acc 47.90%, Band(-12..-6dB) Loss 2.0408 (Acc 22.92%) (best-band)
  Epoch   5: Val Acc 49.92%, Band(-12..-6dB) Loss 1.9834 (Acc 23.59%) (best-band)

  Epoch   6: Val Acc 49.66%, Band(-12..-6dB) Loss 2.0098 (Acc 22.36%)
  Epoch   7: Val Acc 51.13%, Band(-12..-6dB) Loss 1.9027 (Acc 28.15%) (best-band)
  Epoch   8: Val Acc 52.31%, Band(-12..-6dB) Loss 1.9336 (Acc 26.16%)
  Epoch   9: Val Acc 52.67%, Band(-12..-6dB) Loss 1.9056 (Acc 27.70%)
  Epoch  10: Val Acc 54.29%, Band(-12..-6dB) Loss 1.8545 (Acc 29.29%) (best-band)
  Epoch  11: Val Acc 54.91%, Band(-12..-6dB) Loss 1.8485 (Acc 28.96%) (best-band)
  Epoch  12: Val Acc 54.62%, Band(-12..-6dB) Loss 1.8524 (Acc 28.04%)
  Epoch  13: Val Acc 54.52%, Band(-12..-6dB) Loss 1.8221 (Acc 31.06%) (best-band)
  Epoch  14: Val Acc 55.25%, Band(-12..-6dB) Loss 1.8676 (Acc 27.74%)
  Epoch  15: Val Acc 55.30%, Band(-12..-6dB) Loss 1.8062 (Acc 31.11%) (best-band)
  Epoch  16: Val Acc 56.36%, Band(-12..-6dB) Loss 1.7957 (Acc 31.38%) (best-band)
  Epoch  17: Val Acc 54.64%, Band(-12..-6dB) Loss 1.8500 (Acc 29.56%)
  Epoch  18: Val Acc 56.32%, Band(-12..-6dB) Loss 1.7985 (Acc 31.64%)
  Epoch  19: Val Acc 56.44%, Band(-12..-6dB) Loss 1.8000 (Acc 31.08%)
  Epoch  20: Val Acc 56.41%, Band(-12..-6dB) Loss 1.7860 (Acc 31.90%) (best-band)
  Epoch  21: Val Acc 56.42%, Band(-12..-6dB) Loss 1.8168 (Acc 31.51%)
  Epoch  22: Val Acc 56.15%, Band(-12..-6dB) Loss 1.7986 (Acc 31.86%)
  Epoch  23: Val Acc 56.49%, Band(-12..-6dB) Loss 1.7894 (Acc 32.26%)
  Epoch  24: Val Acc 56.52%, Band(-12..-6dB) Loss 1.7915 (Acc 29.59%)
  Epoch  25: Val Acc 56.99%, Band(-12..-6dB) Loss 1.7931 (Acc 32.13%)
  Epoch  26: Val Acc 56.69%, Band(-12..-6dB) Loss 1.7995 (Acc 31.81%)
  Epoch  27: Val Acc 56.66%, Band(-12..-6dB) Loss 1.7819 (Acc 31.56%) (best-band)
  Epoch  28: Val Acc 56.84%, Band(-12..-6dB) Loss 1.7804 (Acc 31.36%) (best-band)
  Epoch  29: Val Acc 56.95%, Band(-12..-6dB) Loss 1.7923 (Acc 31.77%)
  Epoch  30: Val Acc 57.16%, Band(-12..-6dB) Loss 1.8155 (Acc 31.90%)
  Epoch  31: Val Acc 57.02%, Band(-12..-6dB) Loss 1.8012 (Acc 30.85%)
  Epoch  32: Val Acc 56.72%, Band(-12..-6dB) Loss 1.8101 (Acc 29.89%)
  Epoch  33: Val Acc 57.27%, Band(-12..-6dB) Loss 1.7748 (Acc 31.90%) (best-band)
  Epoch  34: Val Acc 57.02%, Band(-12..-6dB) Loss 1.7958 (Acc 31.04%)
  Epoch  35: Val Acc 57.02%, Band(-12..-6dB) Loss 1.8223 (Acc 30.50%)
  Epoch  36: Val Acc 56.88%, Band(-12..-6dB) Loss 1.7825 (Acc 32.80%)
  Epoch  37: Val Acc 57.39%, Band(-12..-6dB) Loss 1.7646 (Acc 31.92%) (best-band)
  Epoch  38: Val Acc 57.03%, Band(-12..-6dB) Loss 1.7663 (Acc 31.81%)
  Epoch  39: Val Acc 57.20%, Band(-12..-6dB) Loss 1.7648 (Acc 31.94%)
  Epoch  40: Val Acc 57.78%, Band(-12..-6dB) Loss 1.7925 (Acc 31.30%)
  Epoch  41: Val Acc 57.75%, Band(-12..-6dB) Loss 1.7853 (Acc 32.88%)
  Epoch  42: Val Acc 57.67%, Band(-12..-6dB) Loss 1.7790 (Acc 32.15%)
  Epoch  43: Val Acc 57.42%, Band(-12..-6dB) Loss 1.7664 (Acc 32.37%)
  Epoch  44: Val Acc 57.55%, Band(-12..-6dB) Loss 1.7789 (Acc 31.02%)
  Epoch  45: Val Acc 57.35%, Band(-12..-6dB) Loss 1.8263 (Acc 30.51%)
  Epoch  46: Val Acc 57.73%, Band(-12..-6dB) Loss 1.7711 (Acc 32.67%)
  Epoch  47: Val Acc 57.61%, Band(-12..-6dB) Loss 1.7898 (Acc 30.93%)
  Epoch  48: Val Acc 56.95%, Band(-12..-6dB) Loss 1.7850 (Acc 31.00%)
  Epoch  49: Val Acc 57.98%, Band(-12..-6dB) Loss 1.7852 (Acc 32.48%)
  Epoch  50: Val Acc 58.05%, Band(-12..-6dB) Loss 1.7695 (Acc 32.54%)
  Epoch  51: Val Acc 58.02%, Band(-12..-6dB) Loss 1.8035 (Acc 31.17%)
  Epoch  52: Val Acc 57.86%, Band(-12..-6dB) Loss 1.7875 (Acc 31.11%)
  Epoch  53: Val Acc 58.41%, Band(-12..-6dB) Loss 1.7674 (Acc 32.05%)
  Epoch  54: Val Acc 58.07%, Band(-12..-6dB) Loss 1.7786 (Acc 31.98%)
  Epoch  55: Val Acc 58.32%, Band(-12..-6dB) Loss 1.7765 (Acc 32.41%)
  Epoch  56: Val Acc 58.12%, Band(-12..-6dB) Loss 1.7822 (Acc 30.87%)
  Epoch  57: Val Acc 58.46%, Band(-12..-6dB) Loss 1.7755 (Acc 32.28%)
  Stopped: no band improvement in 20 epochs (best was epoch 37).
  >> Final test accuracy: 57.97%

============================================================

Config: aug_only  (augmentation=True, batchnorm=False)
Checkpoint selection band: -12 to -6 dB
============================================================
  Trainable parameters: 21,123
  Epoch   1: Val Acc 46.68%, Band(-12..-6dB) Loss 2.0470 (Acc 21.91%) (best-band)
  Epoch   2: Val Acc 50.61%, Band(-12..-6dB) Loss 2.0206 (Acc 23.01%) (best-band)
  Epoch   3: Val Acc 52.12%, Band(-12..-6dB) Loss 1.9595 (Acc 23.61%) (best-band)
  Epoch   4: Val Acc 53.02%, Band(-12..-6dB) Loss 1.9121 (Acc 27.36%) (best-band)
  Epoch   5: Val Acc 53.36%, Band(-12..-6dB) Loss 1.9355 (Acc 25.11%)
  Epoch   6: Val Acc 54.54%, Band(-12..-6dB) Loss 1.8964 (Acc 26.76%) (best-band)
  Epoch   7: Val Acc 54.08%, Band(-12..-6dB) Loss 1.8994 (Acc 26.37%)
  Epoch   8: Val Acc 54.73%, Band(-12..-6dB) Loss 1.9016 (Acc 26.93%)
  Epoch   9: Val Acc 54.93%, Band(-12..-6dB) Loss 1.8721 (Acc 27.96%) (best-band)
  Epoch  10: Val Acc 55.56%, Band(-12..-6dB) Loss 1.8301 (Acc 29.73%) (best-band)
  Epoch  11: Val Acc 56.11%, Band(-12..-6dB) Loss 1.8670 (Acc 28.60%)
  Epoch  12: Val Acc 56.00%, Band(-12..-6dB) Loss 1.8564 (Acc 28.11%)
  Epoch  13: Val Acc 57.45%, Band(-12..-6dB) Loss 1.8457 (Acc 29.35%)
  Epoch  14: Val Acc 57.29%, Band(-12..-6dB) Loss 1.8305 (Acc 29.74%)
  Epoch  15: Val Acc 57.75%, Band(-12..-6dB) Loss 1.8431 (Acc 28.90%)
  Epoch  16: Val Acc 59.31%, Band(-12..-6dB) Loss 1.8375 (Acc 29.84%)
  Epoch  17: Val Acc 59.83%, Band(-12..-6dB) Loss 1.8452 (Acc 29.95%)
  Epoch  18: Val Acc 59.76%, Band(-12..-6dB) Loss 1.8427 (Acc 29.82%)
  Epoch  19: Val Acc 60.19%, Band(-12..-6dB) Loss 1.8651 (Acc 29.09%)
  Epoch  20: Val Acc 60.17%, Band(-12..-6dB) Loss 1.8345 (Acc 29.37%)
  Epoch  21: Val Acc 60.37%, Band(-12..-6dB) Loss 1.8342 (Acc 29.95%)
  Epoch  22: Val Acc 60.68%, Band(-12..-6dB) Loss 1.8462 (Acc 30.16%)
  Epoch  23: Val Acc 60.65%, Band(-12..-6dB) Loss 1.8432 (Acc 29.97%)
  Epoch  24: Val Acc 60.66%, Band(-12..-6dB) Loss 1.8274 (Acc 30.21%) (best-band)
  Epoch  25: Val Acc 60.56%, Band(-12..-6dB) Loss 1.8388 (Acc 30.42%)
  Epoch  26: Val Acc 60.50%, Band(-12..-6dB) Loss 1.8440 (Acc 29.24%)
  Epoch  27: Val Acc 60.59%, Band(-12..-6dB) Loss 1.8393 (Acc 29.73%)
  Epoch  28: Val Acc 60.67%, Band(-12..-6dB) Loss 1.8536 (Acc 30.05%)
  Epoch  29: Val Acc 60.62%, Band(-12..-6dB) Loss 1.8644 (Acc 29.48%)
  Epoch  30: Val Acc 60.71%, Band(-12..-6dB) Loss 1.8505 (Acc 29.80%)
  Epoch  31: Val Acc 60.55%, Band(-12..-6dB) Loss 1.8679 (Acc 29.01%)
  Epoch  32: Val Acc 60.59%, Band(-12..-6dB) Loss 1.8524 (Acc 29.26%)
  Epoch  33: Val Acc 60.44%, Band(-12..-6dB) Loss 1.8303 (Acc 30.10%)
  Epoch  34: Val Acc 60.64%, Band(-12..-6dB) Loss 1.8706 (Acc 29.73%)
  Epoch  35: Val Acc 60.64%, Band(-12..-6dB) Loss 1.8764 (Acc 29.14%)
  Epoch  36: Val Acc 60.61%, Band(-12..-6dB) Loss 1.8737 (Acc 29.14%)
  Epoch  37: Val Acc 60.66%, Band(-12..-6dB) Loss 1.8717 (Acc 29.37%)
  Epoch  38: Val Acc 60.66%, Band(-12..-6dB) Loss 1.8967 (Acc 29.82%)
  Epoch  39: Val Acc 60.72%, Band(-12..-6dB) Loss 1.8911 (Acc 29.33%)
  Epoch  40: Val Acc 60.61%, Band(-12..-6dB) Loss 1.8733 (Acc 30.36%)
  Epoch  41: Val Acc 60.75%, Band(-12..-6dB) Loss 1.8965 (Acc 30.05%)
  Epoch  42: Val Acc 60.51%, Band(-12..-6dB) Loss 1.8812 (Acc 29.37%)
  Epoch  43: Val Acc 60.76%, Band(-12..-6dB) Loss 1.8783 (Acc 30.46%)
  Epoch  44: Val Acc 60.61%, Band(-12..-6dB) Loss 1.9119 (Acc 29.48%)
  Stopped: no band improvement in 20 epochs (best was epoch 24).
  >> Final test accuracy: 61.03%

============================================================
Config: bn_only  (augmentation=False, batchnorm=True)
Checkpoint selection band: -12 to -6 dB
============================================================
  Trainable parameters: 22,011

  Epoch   1: Val Acc 53.88%, Band(-12..-6dB) Loss 1.8836 (Acc 27.68%) (best-band)
  Epoch   2: Val Acc 54.64%, Band(-12..-6dB) Loss 1.8669 (Acc 29.82%) (best-band)
  Epoch   3: Val Acc 55.30%, Band(-12..-6dB) Loss 1.8456 (Acc 28.77%) (best-band)
  Epoch   4: Val Acc 56.97%, Band(-12..-6dB) Loss 1.8011 (Acc 32.05%) (best-band)
  Epoch   5: Val Acc 58.38%, Band(-12..-6dB) Loss 1.7794 (Acc 31.71%) (best-band)
  Epoch   6: Val Acc 59.36%, Band(-12..-6dB) Loss 1.8021 (Acc 31.96%)
  Epoch   7: Val Acc 59.83%, Band(-12..-6dB) Loss 1.7684 (Acc 32.15%) (best-band)
  Epoch   8: Val Acc 59.85%, Band(-12..-6dB) Loss 1.8143 (Acc 31.43%)
  Epoch   9: Val Acc 59.44%, Band(-12..-6dB) Loss 1.7441 (Acc 34.19%) (best-band)
  Epoch  10: Val Acc 60.67%, Band(-12..-6dB) Loss 1.7870 (Acc 32.48%)
  Epoch  11: Val Acc 59.31%, Band(-12..-6dB) Loss 1.7806 (Acc 31.68%)
  Epoch  12: Val Acc 60.48%, Band(-12..-6dB) Loss 1.7706 (Acc 32.46%)
  Epoch  13: Val Acc 60.02%, Band(-12..-6dB) Loss 1.7702 (Acc 32.90%)
  Epoch  14: Val Acc 60.59%, Band(-12..-6dB) Loss 1.7704 (Acc 32.28%)
  Epoch  15: Val Acc 60.76%, Band(-12..-6dB) Loss 1.7802 (Acc 31.68%)
  Epoch  16: Val Acc 60.70%, Band(-12..-6dB) Loss 1.7908 (Acc 32.26%)
  Epoch  17: Val Acc 59.66%, Band(-12..-6dB) Loss 1.7777 (Acc 31.81%)
  Epoch  18: Val Acc 61.02%, Band(-12..-6dB) Loss 1.7926 (Acc 30.96%)
  Epoch  19: Val Acc 61.20%, Band(-12..-6dB) Loss 1.7745 (Acc 32.90%)
  Epoch  20: Val Acc 61.02%, Band(-12..-6dB) Loss 1.7709 (Acc 32.16%)
  Epoch  21: Val Acc 60.45%, Band(-12..-6dB) Loss 1.7786 (Acc 31.86%)
  Epoch  22: Val Acc 61.14%, Band(-12..-6dB) Loss 1.8086 (Acc 30.81%)
  Epoch  23: Val Acc 61.14%, Band(-12..-6dB) Loss 1.8050 (Acc 31.23%)
  Epoch  24: Val Acc 61.28%, Band(-12..-6dB) Loss 1.7759 (Acc 31.73%)
  Epoch  25: Val Acc 61.23%, Band(-12..-6dB) Loss 1.7958 (Acc 31.90%)
  Epoch  26: Val Acc 61.13%, Band(-12..-6dB) Loss 1.7766 (Acc 32.24%)
  Epoch  27: Val Acc 61.17%, Band(-12..-6dB) Loss 1.7740 (Acc 31.88%)
  Epoch  28: Val Acc 60.45%, Band(-12..-6dB) Loss 1.8215 (Acc 31.51%)
  Epoch  29: Val Acc 61.22%, Band(-12..-6dB) Loss 1.7694 (Acc 32.48%)
  Stopped: no band improvement in 20 epochs (best was epoch 9).
  >> Final test accuracy: 59.94%

============================================================
Config: aug_and_bn  (augmentation=True, batchnorm=True)
Checkpoint selection band: -12 to -6 dB
============================================================
  Trainable parameters: 22,011
  Epoch   1: Val Acc 56.69%, Band(-12..-6dB) Loss 1.8848 (Acc 28.71%) (best-band)
  Epoch   2: Val Acc 58.59%, Band(-12..-6dB) Loss 1.9537 (Acc 27.14%)
  Epoch   3: Val Acc 60.36%, Band(-12..-6dB) Loss 1.8460 (Acc 28.90%) (best-band)
  Epoch   4: Val Acc 60.41%, Band(-12..-6dB) Loss 1.8387 (Acc 29.88%) (best-band)
  Epoch   5: Val Acc 60.31%, Band(-12..-6dB) Loss 1.8688 (Acc 28.26%)
  Epoch   6: Val Acc 60.87%, Band(-12..-6dB) Loss 1.8445 (Acc 29.54%)
  Epoch   7: Val Acc 60.64%, Band(-12..-6dB) Loss 1.8561 (Acc 29.26%)
  Epoch   8: Val Acc 60.81%, Band(-12..-6dB) Loss 1.8722 (Acc 29.76%)
  Epoch   9: Val Acc 60.28%, Band(-12..-6dB) Loss 1.8702 (Acc 30.42%)
  Epoch  10: Val Acc 60.65%, Band(-12..-6dB) Loss 1.8836 (Acc 29.09%)
  Epoch  11: Val Acc 60.61%, Band(-12..-6dB) Loss 1.9027 (Acc 28.83%)
  Epoch  12: Val Acc 60.60%, Band(-12..-6dB) Loss 1.8916 (Acc 28.99%)
  Epoch  13: Val Acc 60.72%, Band(-12..-6dB) Loss 1.9062 (Acc 28.32%)
  Epoch  14: Val Acc 59.61%, Band(-12..-6dB) Loss 1.8830 (Acc 29.93%)
  Epoch  15: Val Acc 60.31%, Band(-12..-6dB) Loss 1.9379 (Acc 28.15%)
  Epoch  16: Val Acc 60.49%, Band(-12..-6dB) Loss 1.8945 (Acc 29.14%)
  Epoch  17: Val Acc 60.53%, Band(-12..-6dB) Loss 1.9007 (Acc 28.98%)
  Epoch  18: Val Acc 60.49%, Band(-12..-6dB) Loss 1.9470 (Acc 28.69%)
  Epoch  19: Val Acc 60.13%, Band(-12..-6dB) Loss 1.9447 (Acc 28.53%)

  Epoch  20: Val Acc 60.27%, Band(-12..-6dB) Loss 1.9907 (Acc 28.34%)
  Epoch  21: Val Acc 59.94%, Band(-12..-6dB) Loss 1.9646 (Acc 28.69%)
  Epoch  22: Val Acc 60.22%, Band(-12..-6dB) Loss 1.9860 (Acc 29.01%)
  Epoch  23: Val Acc 60.41%, Band(-12..-6dB) Loss 2.0002 (Acc 28.54%)
  Epoch  24: Val Acc 59.75%, Band(-12..-6dB) Loss 1.9886 (Acc 28.26%)
  Stopped: no band improvement in 20 epochs (best was epoch 4).
  >> Final test accuracy: 61.05%

============================================================
SUMMARY (narrow-band checkpoint selection: -12 to -6 dB)
num_subbands=4, stem_channels=32
============================================================
Config              Params   Test Acc
plain               21,123     57.97%
aug_only            21,123     61.03%
bn_only             22,011     59.94%
aug_and_bn          22,011     61.05%

Config             -20    -18    -16    -14    -12    -10     -8     -6     -4
-2      0      2      4      6      8     10     12     14     16     18
plain            10.2%   9.9%  10.3%  11.8%  16.1%  24.2%  36.7%  50.2%  64.0%  71.6%
80.8%  84.9%  84.6%  85.0%  85.2%  86.7%  85.9%  85.7%  87.3%  86.4%
aug_only         10.3%  10.0%   9.8%  11.8%  13.9%  22.5%  34.9%  50.9%  66.4%  78.6%
86.4%  89.4%  90.2%  91.0%  91.1%  93.1%  92.2%  91.8%  92.5%  91.6%
bn_only          11.1%  10.2%  11.2%  14.3%  18.2%  26.3%  40.2%  52.2%  61.5%  74.7%
80.5%  86.6%  86.6%  88.5%  89.6%  89.0%  89.9%  88.9%  89.1%  88.2%
aug_and_bn       10.5%   9.7%   9.5%  12.1%  14.0%  21.7%  33.1%  52.1%  69.2%  80.0%
86.9%  90.8%  90.5%  90.8%  90.7%  91.8%  92.2%  90.6%  91.9%  90.8%

Script: backend_cut_final

PS E:\Res> python backend_cut_final.py
Using device: cuda

Train samples after augmentation: 598400
backend_cut trainable parameters: 24,171

Epoch   1: Band(-12..-6dB) Loss 1.8761 (Acc 27.59%) (best)
Epoch   2: Band(-12..-6dB) Loss 1.8370 (Acc 29.24%) (best)
Epoch   3: Band(-12..-6dB) Loss 1.8244 (Acc 29.71%) (best)
Epoch   4: Band(-12..-6dB) Loss 1.8464 (Acc 29.48%)
Epoch   5: Band(-12..-6dB) Loss 1.8718 (Acc 29.03%)
Epoch   6: Band(-12..-6dB) Loss 1.8303 (Acc 29.86%)
Epoch   7: Band(-12..-6dB) Loss 1.8324 (Acc 31.06%)
Epoch   8: Band(-12..-6dB) Loss 1.9045 (Acc 28.04%)
Epoch   9: Band(-12..-6dB) Loss 1.8343 (Acc 30.80%)
Epoch  10: Band(-12..-6dB) Loss 1.8497 (Acc 29.80%)
Epoch  11: Band(-12..-6dB) Loss 1.8837 (Acc 29.05%)
Epoch  12: Band(-12..-6dB) Loss 1.8898 (Acc 29.65%)
Epoch  13: Band(-12..-6dB) Loss 1.8576 (Acc 30.50%)
Epoch  14: Band(-12..-6dB) Loss 1.9205 (Acc 29.95%)
Epoch  15: Band(-12..-6dB) Loss 1.9168 (Acc 29.41%)
Epoch  16: Band(-12..-6dB) Loss 1.9314 (Acc 30.10%)
Epoch  17: Band(-12..-6dB) Loss 1.9190 (Acc 29.03%)

Epoch  18: Band(-12..-6dB) Loss 1.9566 (Acc 29.52%)
Epoch  19: Band(-12..-6dB) Loss 1.8966 (Acc 30.91%)
Epoch  20: Band(-12..-6dB) Loss 1.9068 (Acc 30.85%)
Epoch  21: Band(-12..-6dB) Loss 1.9444 (Acc 28.36%)
Epoch  22: Band(-12..-6dB) Loss 1.9573 (Acc 29.46%)
Epoch  23: Band(-12..-6dB) Loss 1.9648 (Acc 28.39%)

Stopped: no improvement in 20 epochs (best was epoch 3).

>> Final test accuracy: 61.04%  (24,171 params)

Accuracy per SNR:
SNR  -20 dB: 10.54%
SNR  -18 dB: 9.31%
SNR  -16 dB: 9.55%
SNR  -14 dB: 11.64%
SNR  -12 dB: 14.63%
SNR  -10 dB: 21.22%
SNR   -8 dB: 33.58%
SNR   -6 dB: 50.70%
SNR   -4 dB: 67.15%
SNR   -2 dB: 77.83%
SNR    0 dB: 85.68%
SNR    2 dB: 90.44%
SNR    4 dB: 90.55%
SNR    6 dB: 92.04%
SNR    8 dB: 91.70%
SNR   10 dB: 92.71%
SNR   12 dB: 92.38%
SNR   14 dB: 91.92%
SNR   16 dB: 93.06%
SNR   18 dB: 91.92%

Script: train_cbn_final

PS E:\Res> python train_cbn_final.py
Using device: cuda

Train samples after augmentation: 598400
HybridAMCNet+ComplexBN trainable parameters: 23,971

Epoch   1: Band(-12..-6dB) Loss 1.8229 (Acc 30.20%) (best)
Epoch   2: Band(-12..-6dB) Loss 1.8996 (Acc 26.37%)
Epoch   3: Band(-12..-6dB) Loss 1.8793 (Acc 29.18%)
Epoch   4: Band(-12..-6dB) Loss 1.8208 (Acc 30.72%) (best)
Epoch   5: Band(-12..-6dB) Loss 1.8293 (Acc 30.83%)
Epoch   6: Band(-12..-6dB) Loss 1.8612 (Acc 29.07%)
Epoch   7: Band(-12..-6dB) Loss 1.8285 (Acc 30.70%)
Epoch   8: Band(-12..-6dB) Loss 1.8602 (Acc 30.05%)
Epoch   9: Band(-12..-6dB) Loss 1.8634 (Acc 29.39%)
Epoch  10: Band(-12..-6dB) Loss 1.8494 (Acc 29.95%)
Epoch  11: Band(-12..-6dB) Loss 1.8772 (Acc 30.36%)
Epoch  12: Band(-12..-6dB) Loss 1.8893 (Acc 29.76%)
Epoch  13: Band(-12..-6dB) Loss 1.8712 (Acc 30.33%)

Epoch  14: Band(-12..-6dB) Loss 1.8811 (Acc 30.70%)
Epoch  15: Band(-12..-6dB) Loss 1.9180 (Acc 29.33%)
Epoch  16: Band(-12..-6dB) Loss 1.9114 (Acc 29.46%)
Epoch  17: Band(-12..-6dB) Loss 1.9311 (Acc 29.48%)
Epoch  18: Band(-12..-6dB) Loss 1.9037 (Acc 30.51%)
Epoch  19: Band(-12..-6dB) Loss 1.9495 (Acc 29.97%)
Epoch  20: Band(-12..-6dB) Loss 1.9857 (Acc 28.60%)
Epoch  21: Band(-12..-6dB) Loss 1.9857 (Acc 28.71%)
Epoch  22: Band(-12..-6dB) Loss 1.9470 (Acc 29.54%)
Epoch  23: Band(-12..-6dB) Loss 2.0123 (Acc 29.13%)
Epoch  24: Band(-12..-6dB) Loss 1.9880 (Acc 28.75%)

Stopped: no improvement in 20 epochs (best was epoch 4).

>> Final test accuracy: 61.33%  (23,971 params)

Accuracy per SNR:
SNR  -20 dB: 10.54%
SNR  -18 dB: 9.44%
SNR  -16 dB: 9.86%
SNR  -14 dB: 11.69%
SNR  -12 dB: 12.75%
SNR  -10 dB: 20.57%
SNR   -8 dB: 35.82%
SNR   -6 dB: 54.15%
SNR   -4 dB: 69.39%
SNR   -2 dB: 80.90%
SNR    0 dB: 86.54%
SNR    2 dB: 90.44%
SNR    4 dB: 90.78%
SNR    6 dB: 91.59%
SNR    8 dB: 90.48%
SNR   10 dB: 92.49%
SNR   12 dB: 91.76%
SNR   14 dB: 91.24%
SNR   16 dB: 92.60%
SNR   18 dB: 91.17%

Script: eval_domain_shift_cspmnet

PS E:\Res> python eval_domain_shift_cspmnet.py
Using device: cuda

Available test conditions:
  clean       : (44000, 2, 128)
  rician_K10  : (44000, 2, 128)
  rician_K3   : (44000, 2, 128)
  rician_K1   : (44000, 2, 128)
  rayleigh_K0 : (44000, 2, 128)
  cfo         : (44000, 2, 128)

CSPMNet parameters: 240,972
Loaded: cspmnet_faithful_best.pt

============================================================
CSPMNet DOMAIN-SHIFT TEST
============================================================
clean          : 62.98%
rician_K10     : 52.32%
rician_K3      : 44.31%
rician_K1      : 38.74%
rayleigh_K0    : 36.46%
cfo            : 50.97%

============================================================
CSPMNet DOMAIN-SHIFT SUMMARY
============================================================
clean           62.98%
rician_K10      52.32% (drop 10.66 pp)
rician_K3       44.31% (drop 18.67 pp)
rician_K1       38.74% (drop 24.24 pp)
rayleigh_K0     36.46% (drop 26.53 pp)
cfo             50.97% (drop 12.01 pp)

Script: eval_domain_shift_ulcnn

PS E:\Res> python eval_domain_shift_ulcnn.py
Using device: cuda

Available test conditions:
  clean       : (44000, 2, 128)
  rician_K10  : (44000, 2, 128)
  rician_K3   : (44000, 2, 128)
  rician_K1   : (44000, 2, 128)
  rayleigh_K0 : (44000, 2, 128)
  cfo         : (44000, 2, 128)

ULCNN parameters: 9,287
Loaded: ulcnn_faithful_best.pt

============================================================
ULCNN DOMAIN-SHIFT TEST
============================================================
clean          : 60.13%
rician_K10     : 49.92%
rician_K3      : 44.80%
rician_K1      : 40.75%
rayleigh_K0    : 39.56%
cfo            : 37.42%

============================================================
ULCNN DOMAIN-SHIFT SUMMARY
============================================================
clean           60.13%
rician_K10      49.92% (drop 10.20 pp)
rician_K3       44.80% (drop 15.33 pp)
rician_K1       40.75% (drop 19.38 pp)

rayleigh_K0     39.56% (drop 20.57 pp)
cfo             37.42% (drop 22.71 pp)

Script: eval domain hybrid_cbn_final

PS E:\Res> python eval_domain_shift.py
Using device: cuda

Available test conditions:
  clean       : (44000, 2, 128)
  rician_K10  : (44000, 2, 128)
  rician_K3   : (44000, 2, 128)
  rician_K1   : (44000, 2, 128)
  rayleigh_K0 : (44000, 2, 128)
  cfo         : (44000, 2, 128)

Model parameters: 23,971
Loaded: hybrid_cbn_final.pt

============================================================
DOMAIN-SHIFT TEST
============================================================
clean          : 61.33%
rician_K10     : 53.13%
rician_K3      : 45.53%
rician_K1      : 39.99%
rayleigh_K0    : 37.67%
cfo            : 50.22%

============================================================
SUMMARY
============================================================
clean           61.33%
rician_K10      53.13%
rician_K3       45.53%
rician_K1       39.99%
rayleigh_K0     37.67%
cfo             50.22%

Script: domain shift ablation_narrowband_aug_and_bn

PS E:\Res> python eval_domain_shift_ablation.py
Using device: cuda

Available test conditions:
  clean       : (44000, 2, 128)
  rician_K10  : (44000, 2, 128)
  rician_K3   : (44000, 2, 128)
  rician_K1   : (44000, 2, 128)
  rayleigh_K0 : (44000, 2, 128)
  cfo         : (44000, 2, 128)

HybridAMCNet parameters: 29,067
Loaded: ablation_narrowband_aug_and_bn.pt
BatchNorm: True

============================================================
HYBRIDAM CNET ABLATION DOMAIN-SHIFT TEST
============================================================
clean          : 62.31%
rician_K10     : 49.73%
rician_K3      : 41.19%
rician_K1      : 35.41%
rayleigh_K0    : 33.48%
cfo            : 51.13%

============================================================
SUMMARY
============================================================
clean           62.31%
rician_K10      49.73% (drop 12.57 pp)
rician_K3       41.19% (drop 21.12 pp)
rician_K1       35.41% (drop 26.90 pp)
rayleigh_K0     33.48% (drop 28.83 pp)
cfo             51.13% (drop 11.18 pp)

============================================================
MODEL INFORMATION
============================================================
Checkpoint       : ablation_narrowband_aug_and_bn.pt
BatchNorm        : True
Subbands         : 8
Parameters       : 29,067

Script: domain shift ablation_narrowband_aug_only

PS E:\Res> python eval_domain_shift_ablation.py
Using device: cuda

Available test conditions:
  clean       : (44000, 2, 128)
  rician_K10  : (44000, 2, 128)
  rician_K3   : (44000, 2, 128)
  rician_K1   : (44000, 2, 128)
  rayleigh_K0 : (44000, 2, 128)
  cfo         : (44000, 2, 128)

HybridAMCNet parameters: 28,059
Loaded: ablation_narrowband_aug_only.pt
BatchNorm: False

============================================================
HYBRIDAM CNET ABLATION DOMAIN-SHIFT TEST
============================================================
clean          : 56.77%
rician_K10     : 44.71%

rician_K3      : 38.58%
rician_K1      : 34.00%
rayleigh_K0    : 32.61%
cfo            : 51.34%

============================================================
SUMMARY
============================================================
clean           56.77%
rician_K10      44.71% (drop 12.06 pp)
rician_K3       38.58% (drop 18.19 pp)
rician_K1       34.00% (drop 22.77 pp)
rayleigh_K0     32.61% (drop 24.16 pp)
cfo             51.34% (drop 5.43 pp)

============================================================
MODEL INFORMATION
============================================================
Checkpoint       : ablation_narrowband_aug_only.pt
BatchNorm        : False
Subbands         : 8
Parameters       : 28,059

Script: domain shift ablation_narrowband_bn_only

PS E:\Res> python eval_domain_shift_ablation.py
Using device: cuda

Available test conditions:
  clean       : (44000, 2, 128)
  rician_K10  : (44000, 2, 128)
  rician_K3   : (44000, 2, 128)
  rician_K1   : (44000, 2, 128)
  rayleigh_K0 : (44000, 2, 128)
  cfo         : (44000, 2, 128)

HybridAMCNet parameters: 29,067
Loaded: ablation_narrowband_bn_only.pt
BatchNorm: True

============================================================
HYBRIDAM CNET ABLATION DOMAIN-SHIFT TEST
============================================================
clean          : 62.12%
rician_K10     : 51.28%
rician_K3      : 42.93%
rician_K1      : 36.96%
rayleigh_K0    : 34.70%
cfo            : 50.97%

============================================================
SUMMARY
============================================================
clean           62.12%

rician_K10      51.28% (drop 10.83 pp)
rician_K3       42.93% (drop 19.19 pp)
rician_K1       36.96% (drop 25.16 pp)
rayleigh_K0     34.70% (drop 27.42 pp)
cfo             50.97% (drop 11.14 pp)

============================================================
MODEL INFORMATION
============================================================
Checkpoint       : ablation_narrowband_bn_only.pt
BatchNorm        : True
Subbands         : 8
Parameters       : 29,067

