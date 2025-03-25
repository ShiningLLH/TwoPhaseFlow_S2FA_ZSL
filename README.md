# Zero-shot state identification for two-phase flow by SD-S2FA
Source code of SD-S2FA on gas-water two-phase flow dataset.
The dataset is obtained through multiphase flow experiment at Tianjin Key Laboratory
of Process Measurement and Control at Tianjin University.

The details of the data and model can be found in    
 [L. H. Li, et al. Zero-Shot State Identification of Industrial Gas–Liquid Two-Phase Flow
 via Supervised Deep Slow and Steady Feature Analysis, TII, 20(6), 8170-8180, 2024.]
(https://doi.org/10.1109/TII.2024.3367045)


#### Notice: 
The code has been modified and improved.
The pre-trained attribute prediction is realized by a shallow embedded network. The final attribute prediction can be achieved by SVM, random forest, LogisticRegression, 
GaussianNB, etc.


#### Fast execution in command line:  
python3 S2FA_ZSL.py  


#### Results Example:  
================= Problem 1: ZSL for Transition States =================  
================= S2FA Siamese Network Training =================  
Epoch 0 | Total_loss: 12.4248 | loss_cov: 3.1619 | loss_slowness: 0.2601 | loss_error:0.2194  
Epoch 10 | Total_loss: 7.9584 | loss_cov: 3.1621 | loss_slowness: 0.1708 | loss_error:0.2503  
Epoch 20 | Total_loss: 7.6801 | loss_cov: 3.1622 | loss_slowness: 0.0902 | loss_error:0.2361  
......  
================= S2FA Siamese Network Testing =================  
Attribute prediction...  
Attribute prediction model: rf  
Flow state identification...  
Average_accuracy: 0.89875  

#### All rights reserved, citing the following papers are required for reference:   
[1] L. H. Li, et al. Zero-Shot State Identification of Industrial Gas–Liquid Two-Phase Flow
via Supervised Deep Slow and Steady Feature Analysis, TII, 20(6), 8170-8180, 2024.