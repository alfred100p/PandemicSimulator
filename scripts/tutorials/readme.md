
# Tutorials

### Table of Contents
<ul>
<li><a href="#setup">Setup</a><br>
<li><a href="#t1">Tutorial 1</a>
<li><a href="#t2">Tutorial 2</a>
</ul>

<h2 id="#setup">Setup</h2>
<ol>
<li> [OPTIONAL] Install Anaconda. Download from <a href="https://www.anaconda.com/">anaconda.com</a>. Try watch this <a href="https://www.youtube.com/watch?v=YJC6ldI3hWk">tutorial</a> if you are getting issues in installation.
<li>[OPTIONAL] After successfull installation of anaconda, create a new anaconda environment. In a terminal/command prompt enter

```shell
conda create --name pansim python=3.8
conda activate pansim
```

You can replace pansim with a name of your choice. <b>Continue using this terminal/command prompt window for the following steps.</b>

<li>Clone the repository this is done by entering the following in a terminal/command prompt window.

```shell
git clone -b tst https://github.com/alfred100p/PandemicSimulator
```

<li>Continue in the same teminal/command prompt. Change current directory and run setup.py

```shell
cd PandemicSimulator
python -m pip install -e .
```
</ol>

Congratulations you finsihed the setup. Time to start the first Tutorial.

<h2 id="#t1">Tutorial 1</h2>

After this tutorial you should get an idea of what this repository is about. You will manually control the stage of response to a simulated pandemic. After that by studying the observations try create a response policy using if-else statements.

<ol>
<li>Run scripts/tutorials/7_run_pandemic_gym_env.py -----NOT SURE IF THIS SHOULD BE IN TUTORIAL. It is a basic run with random actions. 
<li>Run scripts/tutorials/8_manual_control.py This allows you to manually set the stage at each point. Use this to understand which stage to apply at a given momnet by choosing stage based on the observation.
<li>Run scripts/tutorials/9_example_policy_if_else.py This is an example policy implemented using an if else statement based on the observation.
<li>Run scripts/tutorials/10_custom_policy_if_else.py This is another example of a policy with an if else statement. Try replace the statements and create your own policy.
</ol>

<h2 id="#t2">Tutorial 2</h2>

In this tutorial we will work on observstions. An observation is what an agent (your program) can view of the state. Lets consider our current problem of Pandemic Simulation, here we are trying to make a RL agent which will provide us an optimal response to a pandemic. In real life the authorities who provide such response do not have complete knowledge of the state of the environement which would be an accurate number of how many people have the disease. This data is not accurately obtainable because of testing limitations so the data avaialable is the testing data. 
We can also augment the data we get from the observation and add it to the observation to “point out” important information. For example if you consider chess we can make the observation include the position of each piece but you could also include a flag for whether you are in check to point out your king is in danger. This allows us to explicitly convey important information. In our current problem we convey using flags that the current population has more critical cases than the threshold using a flag. In this tutorial we will create another such flag and add it to the observation. 

The data is modelled using a SEIR model. It can be visualised as a pipeline in which the different stages are different places.

Seir model: https://sites.me.ucsb.edu/~moehlis/APC514/tutorials/tutorial_seasonal/node4.html