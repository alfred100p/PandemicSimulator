
# Tutorials

### Table of Contents
<ul>
<li><a href="#setup">Setup</a><br>
<li><a href="#t1">Tutorial 1</a>
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
git clone -b baseNHL https://github.com/alfred100p/PandemicSimulator
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
