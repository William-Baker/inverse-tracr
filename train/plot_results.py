
#%%

# path to events. file in .logs:

# Medium
filepath = ".logs/medium-run GPT2 MEDIUM TASK: Stock LR: 5e-05 TrainFrac:0.5 ParamNoise: 0.0 InpDrop: 0.0 bs: 128 n_embed: 1024 n_layer: 24 n_head: 16/events.out.tfevents.1695831968.gpu-q-29.396893.0"

# Large
# filepath = ".logs/large-run_ii GPT2 LARGE TASK: Stock LR: 5e-05 TrainFrac:0.5 ParamNoise: 0.0 InpDrop: 0.0 bs: 128 n_embed: 1280 n_layer: 36 n_head: 20/events.out.tfevents.1695831971.gpu-q-9.1748145.0"

# 125M
# filepath = ".logs/iTracrV5_1 GPTNEO pythia_125m TASK: Stock LR: 5e-05 TrainFrac:0.5 ParamNoise: 0.0 InpDrop: 0.0 bs: 512 nembed: 768 n_layer: 12 n_head: 12/events.out.tfevents.1695700103.gpu-q-60.1405368.0"


# V2

# large
# filepath = ".logs/large-run_iii GPT2 LARGE TASK: Stock LR: 5e-05 TrainFrac:0.5 ParamNoise: 0.0 InpDrop: 0.0 bs: 128 n_embed: 1280 n_layer: 36 n_head: 20/events.out.tfevents.1695920247.gpu-q-53.3214893.0"

# medium
# filepath = ".logs/medium-run_i GPT2 MEDIUM TASK: Stock LR: 5e-05 TrainFrac:0.5 ParamNoise: 0.0 InpDrop: 0.0 bs: 128 n_embed: 1024 n_layer: 24 n_head: 16/events.out.tfevents.1695920493.gpu-q-28.2272120.0"

# 125M
# filepath = ".logs/neo-run GPTNEO pythia_125m TASK: Stock LR: 5e-05 TrainFrac:0.5 ParamNoise: 0.0 InpDrop: 0.0 bs: 512 n_embed: 768 n_layer: 12 n_head: 12/events.out.tfevents.1695960232.gpu-q-7.1090166.0"


from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
acc = EventAccumulator(
    filepath
)

acc.Reload()

print(acc.Tags())


#'verbose_train/heatmap', 'verbose_train/input_heatmap', 'verbose_val/heatmap', 'verbose_val/input_heatmap', 'verbose_val/input_heatmap_0', 'verbose_val/input_heatmap_1', 'verbose_val/input_heatmap_2', 'verbose_val/input_heatmap_3', 'verbose_val/input_heatmap_4', 'verbose_val/input_heatmap_5', 'verbose_val/input_heatmap_6', 'verbose_val/input_heatmap_7', 'verbose_val/input_heatmap_8', 'verbose_val/input_heatmap_9', 'verbose_val/input_heatmap_10', 'verbose_val/input_heatmap_11', 'verbose_val/input_heatmap_12', 'verbose_val/input_heatmap_13', 'verbose_val/input_heatmap_14'], 
# 'audio': [], 'histograms': ['verbose_train/output', 'verbose_val/output'], 
# 'scalars': ['train_hf/loss', 'train_hf/accuracy', 'train_hf/accuracy100', 'train_hf/accuracy90', 'train_hf/accuracy80', 'train_hf/accuracy70', 'train_hf/accuracy60', 'train_hf/accuracy50', 
# 'verbose_train/acc', 'verbose_val/acc', 
# 'val/accuracy', 'val/accuracy100', 'val/accuracy90', 'val/accuracy80', 'val/accuracy70', 'val/accuracy60', 'val/accuracy50', 
# 'val/loss', 
# 'train_gen/accuracy', 'train_gen/accuracy100', 'train_gen/accuracy90', 'train_gen/accuracy80', 'train_gen/accuracy70', 'train_gen/accuracy60', 'train_gen/accuracy50', 
# 'train_gen/loss', 
# 'eval_single/loss', 'eval_single/accuracy', 'eval_single/accuracy100', 'eval_single/accuracy90', 'eval_single/accuracy80', 'eval_single/accuracy70', 'eval_single/accuracy60', 'eval_single/accuracy50', 
# 'train/loss', 'train/accuracy'], 
# 'distributions': ['verbose_train/output', 'verbose_val/output'


#%%

# CUSTOMISE PLOTS HERE

# Modify the VALUES, NOT the keys, of these dictionaries to change the display names

metrics = {
    'accuracy':'token_accuracy',
    'accuracy70': '70%% accuracy', 
    'accuracy90': '90%% accuracy',
    'accuracy100': '100%% accuracy',
}
# splits = {'train_hf': 'train_forced', 'train_gen':'train_generative',  'eval_single': 'validation_forced', 'val': 'validation_generative'}
splits = {
    'train_hf': 'train forced', 
    'train_gen':'train generative',  
    'eval_single': 'validation forced', 
    'val': 'validation generative',
}
# labels_tv = {'train_hf': 'train', 'train_gen':'train',  'eval_single': 'val', 'val': 'val'}
# labels_fg = {'train_hf': 'forced', 'train_gen':'gen',  'eval_single': 'forced', 'val': 'gen'}



# The number of batches to sample to use to calculate our error bars
peak_range = 500

#%%

peak_scalar = 'val' # scalar from which we'll obtain the step to use for our evalutation
peak_df = pd.DataFrame(acc.Scalars(f"{peak_scalar}/accuracy"))
peak_step = peak_df.iloc[peak_df['value'].argmax()]['step'] # training step to use for our metrics



print(f"Total training steps: {peak_df['step'].max()}")
print(f"Using step {peak_step} for results")





df = []
for split, legend_name in splits.items():
    for metric, metric_name in metrics.items():
        metric_df = pd.DataFrame(acc.Scalars(f"{split}/{metric}"))
        metric_df = metric_df.reset_index() # get arranged integer indices, so iloc == loc
        # value = metric_df[metric_df['step'] == peak_step].iloc[0]['value'] * 100


        # get the index of the step we want to look at
        closest_step = (metric_df['step'] - peak_step).abs().argmin()
        target_index = metric_df.loc[closest_step]
        

        print(closest_step)

        
        


        # take the nearest 100 recordings from the target step (we'll take the mean and std)
        sample = metric_df.iloc[closest_step - peak_range : closest_step + peak_range]

        sample = sample['value'] * 100

        for idx, x in enumerate(sample):

            df.append({'group': legend_name, 'metric': metric, 'Accuracy': x, 'idx': idx})
df = pd.DataFrame(df)

print(df.groupby(['group','metric'])['Accuracy'].mean())

#%%

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
# sns.set_theme(style="whitegrid")



# Draw a nested barplot by species and sex
# g = sns.catplot(
#     data=df, kind="bar",
#     x="group", y="Accuracy", hue="metric",
#     errorbar="sd", palette="dark", alpha=.6, height=6
# )

# plt.figure(figsize=(12, 2.5))
# g = plt.gca()

sns.set_context(rc={"figure.figsize": (20, 6)})
sns.set(rc={'figure.figsize':(20,5)})

rcParams['figure.figsize'] = 20,5

g = sns.catplot(
    data=df, kind="bar",
    x="group", y="Accuracy", hue="metric",
    errorbar="sd", legend=False, height=4, aspect=1.5 #, palette="dark", alpha=.6, height=6
)

g.tick_params(axis='x', rotation=45)
g.despine(left=True)
g.set_axis_labels("", "Accuracy (%)")
# g.legend.set_title("")


plt.legend(loc='upper right')

figpath = f"train/{filepath.split('/')[1][:15]}.pdf"
plt.savefig(figpath, bbox_inches="tight")


# plt.tight_layout()
# s







# %%
