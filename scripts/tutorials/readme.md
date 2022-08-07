
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
<li>[OPTIONAL] After successful installation of anaconda, create a new anaconda environment. In a terminal/command prompt enter

```shell
conda create --name pansim python=3.8
conda activate pansim
```

You can replace pansim with a name of your choice. <b>Continue using this terminal/command prompt window for the following steps.</b>

<li>Clone the repository this is done by entering the following in a terminal/command prompt window.

```shell
git clone -b tst https://github.com/alfred100p/PandemicSimulator
```

<li>Continue in the same terminal/command prompt. Change current directory and run setup.py

```shell
cd PandemicSimulator
python -m pip install -e .
```
</ol>

Congratulations you finished the setup. Time to start the first Tutorial.

<h2 id="#t1">Tutorial 1</h2>

Welcome to the first tutorial.<br>
After this tutorial, you should get an idea of what this repository is about. You will manually control the stage of response to a simulated pandemic. After that by studying the observations try to create a response policy using if-else statements. 

<ol>
<li><b>Read the <a href="https://arxiv.org/abs/2010.10560">Original Paper</a>.</b> <br>This will help you understand the background of this problem and the environment.
<li><b>Run scripts/tutorials/7_run_pandemic_gym_env.py</b>. <br>Do this to understand how to use the environment. 
<li><b>Run scripts/tutorials/8_manual_control.py</b> <br> This allows you to manually set the stage at each point. Use this to understand how the stages affect the case dynamics.
<li><b>Run scripts/tutorials/9_example_policy_if_else.py</b> <br> This is an example policy implemented using an if-else statement based on the observation.
<li><b>Read <a href="#xobs">Explaining the Observation and Action</a></b><br> This will help you understand the observation and make a custom policy for the next part of the tutorial.

<br><br>
<h3 id="#xobs">Explaining the Observation and Action</h3>

<b>Observation Table</b>

---
|Observation|Explanation|How to Access|
| ----------- | -----------| ----------- |
|Critical Population (Testing)|Number of people in Critical condition according to current testing policy|obs.global_testing_summary[...,0]|
|Dead Population (Testing)|Number of people in dead according to current testing policy|obs.global_testing_summary[...,1]|
|Infected Population (Testing)|Number of people in Infected condition according to current testing policy|obs.global_testing_summary[...,2]|
|None Population (Testing)|Number of people in not in any other condition according to current testing policy|obs.global_testing_summary[...,3]|
|Recovered Population (Testing)|Number of people in Recovered condition according to current testing policy|obs.global_testing_summary[...,4]|
|Critical Population (Actual)| Actual number of people in Critical condition|obs.global_infection_summary[...,0]|
|Dead Population (Actual)|Actual number of people in dead |obs.global_infection_summary[...,1]|
|Infected Population (Actual)|Actual number of people in Infected condition |obs.global_infection_summary[...,2]|
|None Population (Actual)|Actual number of people in not in any other condition |obs.global_infection_summary[...,3]|
|Recovered Population (Actual)|Actual number of people in Recovered condition |obs.global_infection_summary[...,4]|
|Current Stage|Stage of Response at Current timestep|obs.stage[...,0]|
|Infection Flag|Whether Number of Infected People according to current testing policy exceeds threshold specified according to current simulator configuration (10 by default)|obs.infection_above_threshold[...,0]|
|Current Day|Current Day of simulation|obs.time_day[...,0]|
|Current Unlocked Non Essential Business Locations|List of ids of businesses that are Unlocked (By default additional businesses are not used in the simulator, so this is unused) |obs.unlocked_non_essential_business_locations[...,0]|
---

<b>Action  Table</b>

---
|Stages|Stay home if sick, Practice good hygiene| Wear facial coverings| Social distancing| Avoid gathering size (Risk: number)|Locked locations|
| ----------- | -----------| ----------- | ----------- | -----------| ----------- |
|Stage 0| False|False|None|None|None|
|Stage 1| True|False|None|Low: 50, High: 25|None|
|Stage 2| True|True|0.3|Low: 25, High: 10|School, Hair Salon|
|Stage 3| True|True|0.5|Low: 0, High: 0 |School, Hair Salon|
|Stage 4| True|True|0.7|Low: 0, High: 0 |School, Hair Salon, Office, Retail Store|

---

<br><br>

<li><b>Run scripts/tutorials/10_custom_policy_if_else.py </b><br> This is another example of a policy with an if-else statement. Try to replace the statements and create your own policy.
<li><b>Run scripts/tutorials/11_custom_policy_test.py </b><br> Take your policy from scripts/tutorials/custom_policy_if_else.py and run this script to evaluate your policy over multiple episodes.
</ol>

<h2 id="#t2">Tutorial 2</h2>

In this tutorial we will work on modifying the observation. An observation consists of the variables your policy is allowed to use to make decisions. Let's consider the pandemic response problem: here, we are trying to design a policy that would provide an optimal decision/resposne to the current state of the pandemic. In real life, the authorities who provide such a response may not have complete knowledge of the state of the environment---for instance, an accurate count of how many people truly have the disease, rather than the proportion of people who tested positive. If we could give this crucial information to the authorities, their decision-making could improve dramatically.

This tutorial shows how to augment the observation with additional variables, so that we may create policies that explicitly consider this additional information. Specifically, we will create a flag indicating whether the current population has more critical cases than a certain threshold and add it to the observation.

ALL LINE NUMBERS IN THE FOLLOWING REFER TO THE INITIAL STATE OF THE CODE. 
PLEASE NOTE LINE NUMBER VALUES MAY CHANGE IF ADDITIONAL CHARACTERS OR LINES ARE ADDED WHILE CODING. 


Steps
<ol>
<li><b>Review <a href="#xobs">Explaining the Observation and Action</a></b>
<li><b> Add Critical FLag </b><br>
To modify the observation we can change the code at two levels. We can either change the structure of the simulator state or we can modify the final observation. For this tutorial we will modify the simulator state. The code for simulator state is in python/pandemic_simulator/environment/interfaces/sim_state.py

For demonstration, we will be creating a new flag variable for the environment. This flag will check if the number of patients who are critical exceeds a threshold. The name of the variable should be `critical_above_threhsold`.
<ol>
<li><b>Modify simulator state template to add new flag.</b><br>
 Open python/pandemic_simulator/environment/interfaces/sim_state.py
(Line 52) Add code for the new flag in the specified area. This change will allow the sim_state to hold the new flag. 

<li><b>Modify simulator config to add threshold, as it is a characteristic of the simulator</b><br>
 Open python/pandemic_simulator/environment/simulator_opts.py 
(Line 49) Add code for a threshold to use for the flag and assign a default value of 10. The threshold is not a variable in the PandemicSimulator class
as it is constant throughout the simulation and is a characteristic of the simulator.

<li><b>Write code to incorporate the flag in the simulator</b><br>
Open python/pandemic_simulator/environment/pandemic_sim.py 
(Line 61,91,119) Here, we will have to modify the init function. This is to enable users to add a value for our new flag when initializing the 
Pandemic Simulator State.

(Line 166) Change from_config function

(Line 323) Next we will have to modify the step function in pandemic_sim.py. This is to add code to update the simulator state after each timestep (hour).

(Line 414) Then set a default value for flag in the reset function.

<li><b>Modify the observation for flag</b>
Open python/pandemic_simulator/environment/interfaces/pandemic_observation.py
(Line 25) Add code to initialize flag in observation.
(Line 44) Add code to attribute for flag in function.
(Line 78) Add code to update flag in observation.
</ol>

<li>Add beta Using Testing data into the observation, considering a time period of 1 day.<br>
<ul>
<li>beta(t)=[None(t-1)-None(t)]/[None(t)*Infected(t)]. 
<li>The variable should be accessible via obs.beta
</ul>
</ol>
