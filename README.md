# t-pred
A  repository that stores multiple models for T cell epitope binding prediction

You can start with conda like this:

1: Clone the repository
~~~bash
cd goal_dic
git clone https://github.com/LuoMole/t-pred.git
~~~
2: Create your env
~~~bash
conda env create -f environment.yml -n project_name
~~~
3: You can directly run the training process by
~~~bash
python train_FNNemb.py # take the FNNemb model for example
~~~
Or you can get some help by
~~~
python train_FNNemb.py -h 
~~~
so that you can figure out the meaning of each parameter

4: Once you finish the training process, you can run the test process
~~~bash
python test_FNNemb.py  
~~~
5: Both train and test result will be stored in the "fig" folder
such as
![](https://github.com/LuoMole/t-pred/blob/main/model_better/bert/loss_curve_train.png)
![](https://github.com/LuoMole/t-pred/blob/main/model_better/bert/auc_val.png)


6: To get access to the GUI, please download the GUI.exe, which is available at https://1drv.ms/u/c/d530cdfc41614d1d/ERVdeEDAeMJDmdOGorG7nWUBGPC-7PMoP6loHxaJCWhtiw?e=yvufmM .

Please ensure that the exe and model files are under the same folder.

After submitting the relative path, TCR, and epitope data of the model, click the "提交" button to obtain the prediction results.
！！！！！！
Gpu is necessary for your device to run the GUI
！！！！！！


Following are some usage scenarios of the GUI.

![](https://github.com/LuoMole/t-pred/blob/main/fig/af4010e7e0cb7f2c8563f07281a3032.png)
![](https://github.com/LuoMole/t-pred/blob/main/fig/e76d08cfb9ec56baa49b4a4ea75654c.png)
