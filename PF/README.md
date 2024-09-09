# Model-Based State Estimation with a Particle Filter
The two folders differ primarily because of how the data files are formatted. 

"Real Data" works with data from the UNM study, while "Simulated Data" is the synthetic data generated with the scripts in *../Synth/*.

Since the synthetic data is not related to any humans, it is included in the folder directly. For the real data, anyone who's interested can contact the Julius Lab.

### Pure Predictions
We compare the state estimator with pure predictions made with the same underlying JFK model. To generate those predictions, use *Generate_Sim_Predictions.m*.

### State Estimator
To run the state estimation, we need to first optimize the Kalman filter on the subject's data using ***optimize_kf.m***. The script optimizes the filter, then saves the filter outputs yHat and phi to a .mat file. Once that file has been generated, we can then use ***run_particle_filter_delta_tau.m*** to perform the state estimation with the particle filter.



#### Citation
``` bibtex
@article{ike2024model,
  title={Model-Based Human Circadian State Estimation with Wearable Device Data},
  author={Ike, Chukwuemeka O and Wen, John T and Oishi, Meeko MK and Brown, Lee K and Julius, A Agung},
}
```