# Running Instructions for Grokking Project (Project Group 23)

The project involves 4 notebooks:

> Group23-grok_train.ipynb (for training and checkpointing of a grokking model)  
> Group23-grok_sweep.ipynb (for performing a sweep over different training fractions and weight decays, for different seeds)  
> Group23-grok_analysis_noncausal.ipynb (for running an analysis with bidirectional attention)  
> Group23-grok_analysis_causal.ipynb (for running an analysis with masked attention -> causal model)  

### 1) Minor package installation required (not included in EECS JupyterHub environment)

Before running any notebook, it is required to run the following command in a Terminal to install the required project dependencies (apart from the ones included in the notebook import sections which were already installed in the EECS JuypterHub environment):

`pip install plotly anywidget`

After having installed these two Python libraries into the respective Python execution environment, please refresh the browser page and restart the kernels that were possibly running already. Only then the live dashboarding in the training notebook will display and refresh correctly.


### 2) Grid search 

The notebook already contains the results of the hyperparameter sweep run. It is however not recommended to run this notebook as the grid search took approximately 18.2 hours.

### 3) Analysis notebooks

The only noteworthy notes to mention here is that the user might want to modify the `KEY_STEPS` variable which appears several times across the respective causal/non-causal notebook depending on the wanted granularity of the quantitative and qualitative results. Also, it is important to note that the user needs to specify the frequencies `FREQS` they are interested in for the linear probing of the Fourier components if they do not want the analysis to run for all frequencies. We apologise for this intermediate hard-coding step, but the conducted analyses were already excessive so we wanted to provide the options to narrow down the analysis granularity.