# mGRN
Memory-Gated Recurrent Networks (AAAI 2021)

Appendices: aaai_supplement.pdf

### Requirements
Python == 3.6. Pytorch == 1.5. By default, the codes and the pre-trained models run on GPUs. 

### Simulation Experiments
Simulated data is stored in *simulation_data/data_generation_matlab/simulation_data* in the CSV format. The first two columns in each csv file contain the observations of $y_1$ and $y_2$. The next seven columns are the parameters of $y_1$. The last seven columns are the parameters of $y_2$. Theoretical minimum MSEs are saved in *simulation\_data/data\_generation\_matlab/simulation\_data\_mse.csv*. The Matlab code used to generate the simulated data and the theoretical MSEs can be found in 
 *simulation\_data/data\_generation\_matlab/simulation\_ar\_main.m*. 
 
To train and evaluate new models, please refer to *NN\_main\_simulation\_regression.py*. To validate pre-trained models, please refer to *validation\_regression.py*. To reproduce table 1 and figure 2 in the paper, please refer to *simulation\_results\_summary.ipynb*. 

### MIMIC-III Data Set
Please download the MIMIC-III data set, and follow the instructions in the [mimic-3 repository](https://github.com/YerevaNN/mimic3-benchmarks) to preprocess the data. You should finish all the steps in Section *Building a benchmark* and *Train / validation split*. After these steps, you should be able to run *main.py* of the four tasks in the mimic-3 repository.

To accelerate training, we save the processed data into the NPZ format. We are not able to directly provide these files due to the constrained usage of the MIMIC-III data set. Please run *{task}_data_preparation.py* in folder *mimic3\_utils* for each task. A folder named by the task will be created under *mimic3\_utils*. For example, for the in-hospital mortality task, please run *mimic3\_utils/ihm\_data\_preparation.py*. Note that these files make use of multi-processors by default and are time-consuming. 

To train new models, please refer to *NN\_main\_mimic3_{task}.py* of each task. For example, for the in-hospital mortality task, please run *NN\_main\_mimic3\_ihm.py*. The hyper-parameters used to obtain results in the paper are saved in the respective *NN\_main\_mimic3\_\{task\}.py* of each task. We repeat each experiment three times and the result with the best validation performance is reported. To validate pre-trained models, please run *validation\_mimic3\_{task}.py* of each task.

### UEA Data Sets
The UEA data sets can be downloaded [here](http://www.timeseriesclassification.com/dataset.php). You may run the *uea\_utils/uea\_dataset\_preparation.py* to automatically download and preprocess the data sets. Folders named by data sets will be created under *uea\_utils*.

The hyper-parameters used to obtain results in the paper are saved in *uea\_utils/param\_dict.py*.
To train and evaluate new models, please refer to *NN\_main\_uea.py*.  We repeat each experiment three times and the best result is reported. To validate pre-trained models, please refer to *validation\_uea.py*.
