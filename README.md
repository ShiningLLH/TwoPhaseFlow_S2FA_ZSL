# TwoPhaseFlow_S2FA_ZSL

# Zero-shot state identification for two-phase flow by SD-S2FA
Source code of SD-S2FA on gas-water two-phase flow dataset.
The dataset is obtained through multiphase flow experiment at Tianjin Key Laboratoryof Process Measurement and Control at Tianjin University.

The details of the model can be found in    
 [L. H. Li, et al. Zero-Shot State Identification of Industrial Gas–Liquid Two-Phase Flow via Supervised Deep Slow and Steady Feature Analysis, TII, 20(6), 8170-8180, 2024.]
(https://doi.org/10.1109/TII.2024.3367045)

#### Notice: 
The code has been modified and improved.
The pre-trained attribute prediction is realized by a shallow embedded network.
It makes the SD-S2FA network more lightweight.
The final attribute prediction can be achieved by SVM, random forest, LogisticRegression, GaussianNB, etc.

#### Fast execution in command line:  
python3 S2FA_ZSL.py      

#### Results Example: 
================= Problem 1: ZSL for Transition States =================   
================= S2FA Siamese Network Testing =================   
Attribute prediction...  
Attribute prediction model: SVC_rbf  
Flow state identification...  
Average_accuracy: 0.9203268641470889  

================= Problem 2: ZSL for Unknown Typical States =================  
================= S2FA Siamese Network Testing =================  
Attribute prediction...  
Attribute prediction model: lr  
Flow state identification...  
Average_accuracy: 0.9192229038854806  

================= Problem 3: ZSL for Typical States in Unknown Working Conditions =================  
================= S2FA Siamese Network Testing =================  
Attribute prediction...  
Attribute prediction model: SVC_rbf  
Flow state identification...  
Average_accuracy: 0.8526208304969367  

#### All rights reserved, citing the following papers are required for reference:   
[1] L. H. Li, et al. Zero-Shot State Identification of Industrial Gas–Liquid Two-Phase Flow via Supervised Deep Slow and Steady Feature Analysis, TII, 20(6), 8170-8180, 2024.
