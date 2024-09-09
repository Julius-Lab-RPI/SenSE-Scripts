To generate a synthetic subject's data, run the following scripts in order:
1. Generate_Sim_Light
2. Generate_Sim_Data - first modify jewett99's tau in "..\\..\Models"
3. generate_basal_hr - pass the subject's Sim_Data filename. Takes 10-20 minutes to run for a 15-day trajectory.
4. Generate_Sim_Activity
5. Generate_Sim_Cortisol
6. generate_sim_hr - runs per subject and appends HR to Data\Sim_A*.mat.

Each script contains documentation on the available parameters to choose.

The generation scripts visualize the data as they're generated, but to run just the visualizations, use the following:
1. Plot_Subject_Data - plots true JFK, CBTmins, HR
2. Plot_Dataset - plots all subjects: JFK, CBTmins, HR