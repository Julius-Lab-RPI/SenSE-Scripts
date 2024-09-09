Instead of repeating the scripts that are in ../KFSS/, this folder just contains ML/ which holds Yunshi Wen's code implementing the AE + GRU architecture used in doing the system identification. The folder also has the data files and specific saved results from the EMBC 2024 paper.

To replicate the work in the paper, one would first need to optimize the Kalman filter from ../KFSS/ on a specific subject's data, the run it on their data and save the filtered output yHat and phase output phi. The phi is what is then used in Yunshi's training as the similarity label. If this sounds confusing, feel free to ask either Chukwuemeka Ike or Yunshi Wen.


**Note: The contents of ML/ are the work of Yunshi Wen, and should be attributed accordingly.**