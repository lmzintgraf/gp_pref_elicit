This is the code for the paper

> Ordered Preference Elicitation Strategies for Supporting Multi-Objective Decision Making  
> Luisa M Zintgraf, Diederik M Roijers, Sjoerd Linders, Catholijn M Jonker, Ann Nowe  
> _17th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2018)_

which you can find on [arXiv](https://arxiv.org/abs/1802.07606).

## Main Implementation

The main code, the implementation of pairwise gaussian processes, 
is comprised of `gaussian_process.py`, `acquisition_function.py`, and `dataset.py`.  

For examples on how to use them, we refer to `gp_utilities/utils_experiment.py` and to the `experiments` folder.
There you can also find all experiments we ran for the paper.
If you want to run something and get pretty plots, try
```
python experiments/exp_gp-shape.py
```

## Web Interface

The source code for the user study can be found in the folder `webInterface`.
To run it, type
```
python webInterface/start.py
```
in a terminal and go to `http://0.0.0.0:5000/` in a web browser of your choice 
(I tested a few, but can't guarantee it works everywhere).
There you should find the starting page of the experiment. 
Flags: `-t` to skip the tutorial, `-d` for debug mode.