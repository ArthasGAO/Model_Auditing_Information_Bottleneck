Selection plans for run_hypothesis_test_at_traj.py.

Each *.yaml here declares ONE Scenario_Name: the `Scenario` column that
calculate_MI_at_traj.py wrote for one post-hoc AT source (the alias / base
model name minus its trailing _<seed>_<rate>). They select rows of
saved_logs/at_evasion/MI_master_table_at_traj.csv; the driver's `filters`
(run_tag, ckpt_kind) then pick the run and the epoch. Nothing here is a
training plan -- the training plans are saved_exp_plan/at_plan/
CIFAR10_RES18_PostAT_MixOff_{FTAL_Prune20,DKD_Knockoff}.yaml.
