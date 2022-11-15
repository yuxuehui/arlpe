# ARLPE: A Meta Reinforcement Learning Framework for Glucose Regulation in Type 1 Diabetics

This is the official implementation of ARLPE (accepted to Expert Systems with Applications), which achieves the automatic glucose regulation of unknown patients even with uncertainties and noise. The performance of ARLPE is evaluated in 30 *in silico* patients from the Diabetes Mellitus Metabolic Simulator for Research (DMMS.R) [<sup>1</sup>](#refer-anchor-1), a computer application designed for clinical trials in virtual subjects and certified by the US FDA.

<img src="https://raw.githubusercontent.com/yuxuehui/arlpe/main/fig/teg-logo.png" width="50%" style="display:block; margin:0 auto;">


## Abstract
External artificial pancreas (AP) with autonomous control algorithms has proved its effectiveness in glucose regulation for type 1 diabetes (T1D). Nonetheless, most existing algorithms cannot adapt to unknown patients with limited clinical data. To achieve the automatic glucose regulation of unknown patients even with uncertainties and noise, we propose an Active Reinforcement Learning with Personalized Embeddings (ARLPE) for normoglycemia maintenance. Our framework contains a meta-training period and a fine-tuning period. The meta-training period aims to learn: 1) a generalized policy for glucose regulation and 2) a probabilistic encoder that summarizes the personalized information and context into an embedding. The fine-tuning period is designed to generate a personalized policy for the unknown patient with the help of an active learning module to explore valuable experiences. Experiments on multiple patients demonstrate that our algorithm can not only converge blood glucose (BG) to the normoglycemic bounds and avoid hypoglycemia (Hypo) with a time in range (TIR) of 98.63% but also achieve the glucose regulation of a new unacquainted patient using limited BG data (only 25 samples). Compared with other state-of-the-art algorithms, ARLPE significantly outperforms the best competing methods for glucose regulation. It shows the great potential of generating personalized clinical strategies for diabetics.

<img src="https://raw.githubusercontent.com/yuxuehui/arlpe/main/fig/model.png" width="100%" style="display:block; margin:0 auto;">

</br>

## Instructions

The entire communication framework is shown as follow.

<img src="https://raw.githubusercontent.com/yuxuehui/arlpe/main/fig/test_pig.png" width="80%" style="display:block; margin:0 auto;">

IDE version:

```python
Python: 3.8.5
Django: 2.0
PyMySQL: 1.0.2
MySQL: 8.0.22-0ubuntu0.20.04.3
```

**(1) Installation of DMMS.R Application**

First, the DMMS.R application needs to be purchased and installed. Available operating systems include: Windows 7 (64-bit), Windows 8 (64-bit), or Windows 10. Please contact Epsilon by e-mail for more information: mlEpsilonInfo@abbott.com

DMMS.R GUI is start by runing `dmmsDjango/RLType1/rlkit/samplers/cmdcontrol.py`. And you need to configure the IP address and username and password of the computer where the DMMS.R is located.

```python
DMMS_ip = "" # line 9

s = winrm.Session(DMMS_ip, auth=('XXX', 'XXX')) # line 45
r = s.run_cmd('XXXX') # line 46
```

**(2) Deploy Django**

We write a JavaScript file to produce custom elements (sensor, control element, and delivery element). Specifically, we use Python to write the JavaScript file, taking advantage of the JavaScript plugins' support for web service calls. The web service is built through a Django web framework. Therefore, before running ARLPE in the third step, you must run the client and server and successfully establish a connection. Django can be started by running the python file "/arlpe/manage.py":

```python
python manage.py runserver 0.0.0.0:<your port>
```

0.0.0.0 can be accessed from any origin in the network. For port number, you need to set inbound and outbound policy of your system if you want to use your own port number not the default one.


**(3) Deploy MySQL**

we use MySQL in our Django web framework and config Django using pymysql as driver. And MySQL is configured via json configuration files `dmmsDjango/RLType1/configs/mysql_config.py`

**(4) ARLPE**

Our framework contains a meta-training period and a fine-tuning period. The meta-training period aims to learn: 1) a generalized policy for glucose regulation and 2) a probabilistic encoder that summarizes the personalized information and context into an embedding.

1) Meta-training period

The initial BG level for each virtual patient was set as the default, which was relatively high. The agent must control the BG level within the normoglycemic range as quickly as possible. In other words, the agent should accomplish the glucose regulation in 72h after admission and maintain it within the normoglycemic range during the following days. During meta-training, the max path length of an episode for the environment was 60 simulated days, and the generalized model trained based on Algorithm 1.

<img src="https://raw.githubusercontent.com/yuxuehui/arlpe/main/fig/meta_learning.png" width="50%" style="display:block; margin:0 auto;">

Experiments are configured via json configuration files located in ./configs. To reproduce an experiment, run: 
```python
python 0_meta_training.py ./configs/[EXP].json
```

By default the code will use the GPU - to use CPU instead, set `use_gpu=False` in the appropriate config file.

Output files will be written to `./output/[ENV]/[EXP NAME]` where the experiment name is uniquely generated based on the date.


2) Fine-tuning period

The generalized model was fine-tuned for the testing task through the active learning approach presented in Algorithm 2. All models were tested over 30 days, excluding the first day. That is because insulin on board was zero at the start of each trial, which might cause hyperglycemia on the first day.

<img src="https://raw.githubusercontent.com/yuxuehui/arlpe/main/fig/meta_testing.png" width="50%" style="display:block; margin:0 auto;">


To test and fine-tune an optimal generalized model, run the following:
```python
python 0_active_learning.py
```

## Parameter Selection Suggestions
We provide several parameter selection suggestions for the following reward function and corresponding analyses.

$$
r(G) = \beta - dist_{outside}(G;\textbf{q}) - \alpha \cdot dist_{inside}(G;\textbf{q})
$$

where $G$ is the query, $\alpha=0.2$, $\beta=24$, and $\textbf{q}=\left[ q_{min}, q_{max} \right] \in \mathbb{R}^{2d}$ is a query box that is $\left[ 70,180 \right]$ in the problem of glucose regulation. In this project, both $\alpha$ and $\beta$ use the optimal values provided by Ren [<sup>1</sup>](#refer-anchor-2) and can be adjusted according to different task demands. The visualization of above function is shown as follows:

<img src="https://raw.githubusercontent.com/yuxuehui/arlpe/main/fig/reward_va.png" width="50%" style="display:block; margin:0 auto;">


Moreover, other parameter selection suggestions are shown in the following.

Hyperparameter $\alpha$ is a discount factor that adjusts the attention of $dist_{outside}$ and $dist_{inside}$ in equation (7). $dist_{outside}$ penalizes agents with negative value while blood glucose levels fall outside the target range; $dist_{inside}$ awards agents while blood glucose levels fluctuate around $Cen(\textbf{q})$ and keep away from the boundary. The following figure shows the comparative analysis of different $\alpha$. As the value of $\alpha$ becomes larger, the range of the highest reward (the red area) gradually shrinks, and the blood glucose value approaches the median $Cen(\textbf{q})$ of the normal range $\textbf{q}$. It should be noted that the increase of $\alpha$ will also cause difficulty for agents exploring the highest reward area.


<img src="https://raw.githubusercontent.com/yuxuehui/arlpe/main/fig/test_alpha.png" width="100%" style="display:block; margin:0 auto;">

Hyperparameter $\beta$ can adjust the value of the highest reward. Figure \ref{reward_function_beta} shows the comparative analysis of different $\beta$. With the $\beta$ increasing, the highest reward value would increase. Similarly, the increase of $\beta$ will also increase the difficulty of agent learning.

<img src="https://raw.githubusercontent.com/yuxuehui/arlpe/main/fig/test_beta.png" width="100%" style="display:block; margin:0 auto;">

## 参考
<div id="refer-anchor-1"></div>
[1] [DMMS.R - The Epsilon Group](https://tegvirginia.com/software/dmms-r/)

<div id="refer-anchor-2"></div>
[2] [Query2box: Reasoning over Knowledge Graphs in Vector Space using Box Embeddings](https://doi.org/10.48550/arXiv.2002.05969)