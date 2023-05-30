[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![GPLv3 license](https://img.shields.io/badge/License-GPLv3-blue.svg)](http://perso.crans.org/besson/LICENSE.html)


# Constraint-Based Scenario for Team Formation Problems Based on Intelligent Virtual Agents

We have designed a testing scenario that accommodates the formation of both regular and robust teams.

The cornerstone of our work lies in its capability to dynamically generate testing environments from the context of team formation problems. Following the [Okimoto](https://scholar.google.com/citations?user=QcGxbbkAAAAJ&hl=en)'s definition, a team formation problem, `TF`, is denoted as `⟨A, P, f, α⟩`, representing agents, tasks, costs, and the assignment of tasks each agent can complete. We detail the specifics of the team formation problem in a JSON file to define a team formation problem.

Once the JSON file is created, the simulator then translates this into the necessary environments. Upon creating the environment, we execute the simulations on the [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator) a framework develop by [Matteo Bettini](https://matteobettini.com) and [Prolok Lab](https://www.proroklab.org/) at [University of Cambridge](https://www.cst.cam.ac.uk). The VMAS framework currently supports both heuristic and RLlib for agent control. This design, which capitalizes on the robustness of VMAS, offers a versatile and comprehensive testing ground for various scenarios pertaining to Multi-Agent Reinforcement Learning.


## Supported VMAS Environments
We have chosen several scenarios that optimally represent team formation problems, providing a testbed for robustness.
| Heuristic | RLlib|
|:-------------------------:|:-------------------------:|
|<img width="1604" src="https://github.com/matteobettini/vmas-media/blob/main/media/scenarios/transport.gif?raw=true">  Transport |  <img width="1604" src="https://github.com/matteobettini/vmas-media/blob/main/media/scenarios/balance.gif?raw=true"> Balance|
|<img width="1604" src="https://github.com/matteobettini/vmas-media/blob/main/media/scenarios/reverse_transport.gif?raw=true"> Reverse Transport |  <img width="1604" src="https://github.com/matteobettini/vmas-media/blob/main/media/scenarios/ball_trajectory.gif?raw=true"> Ball Trajectory|
|<img width="1604"  src="https://github.com/matteobettini/vmas-media/blob/main/media/scenarios/wheel.gif?raw=true">  Wheel |


## Structure of the JSON Simulation File
The JSON file consists of four main sections:

- `settings` is used for determining the global settings of the simulation.
- `agents` involves a list where each kind of agent is described, including their respective skills, i.e., tasks they can complete.
- `teams` introduces a list of different teams, formed by a mix of the agents described earlier.
- `goals` outlines a list of goals; each goal can include multiple tasks (scenarios). In these scenarios, teams are evaluated to check if they have agents equipped with the skills required for the specified task.

```json
{
  "settings": {},
  "agents": [],
  "teams": [],
  "goals": []
}
```

Now, let's explore each section in more detail.

#### Settings
The following settings apply to all goals and their tasks/scenarios:
- `version` Defines the version of the current JSON structure. Its purpose is to facilitate version management, particularly when there are structural changes that break compatibility with previous versions.
- `name` Specifies the name of the simulation, primarily for logging purposes.
- `render` Indicates that each execution can produce a video.
- `save` Suggests that the video created can be saved.
- `failure_probability` Defines the likelihood of an agent failing at any given step. It will not apply once k_robustness agents have failed, or when k_robustness is 0 or undefined.
```json
{
  "settings": {
    "version": "1.0"
    "name": "Simulation 1",
    "render": true,
    "save": true,
    "failure_probability": 0.1
  },
  "agents": [],
  "teams": [],
  "goals": []
}
```

#### Agents
- `id` A unique identifier that will be used to construct the teams.
- `cost` Represents the general resource consumption associated with using these agents.
- `skills` A list indicating the types of tasks that the agents can complete. Currently, the supported values for `skills` include `transport`, `reverse_transport`, `wheel`, `balance`, and `ball_trajectory`.
```json
{
  "settings": {},
  "agents": [
    {
      "id": "agent1",
      "cost": 4,
      "skills": [
        "transport",
        "balance"
      ]
    }
    ],
  "teams": [],
  "goals": []
}
```

#### Teams
- `id` A unique identifier, primarily for logging purposes.
- `k_robustness` indicates how many agents can fail and yet being able to complete the tasks.
- `agents` a list of the agent's ide that were defined on `agents`, it's allowed to use multiple agents of the same type.
```json
{
  "settings": {},
  "agents": [],
  "teams": [
    {
      "id": "team1",
      "k_robustness": 1,
      "agents": [
        "agent1",
        "agent1"
      ]
    }
  ],
  "goals": []
}
```

#### Goals
- `id` A unique identifier, primarily for logging purposes.
- `tasks`: A goal requires the completion of one or more tasks, each of which is defined as follows:

    ##### Task
  - `environment`: This refers to the VMAS environment to be executed. The currently supported environments include `transport`, `reverse_transport`, `wheel`, `balance`, and `ball_trajectory`.
  - `env_kwargs`: These are custom parameters that vary depending on the environment. The supported parameters will be specified in a subsequent section, but in general, they adhere to a `key`:value structure.
```json
{
  "settings": {},
  "agents": [],
  "teams": [],
  "goals": [
    {
      "id": "goal1",
      "tasks": [
        {
          "environment": "transport",
          "env_kwargs": {
            "n_packages": 1,
            "package_width": 0.2,
            "package_length": 0.2,
            "package_mass": 10
          }
        }
      ]
    }
  ]
}
```

## Examples
##### Robust Team in the `transport` Environment
Here is an example involving a team of four identical agents capable of executing `transport` tasks. This team maintains a robustness of `k=2`, meaning up to two agents can fail while the team can still complete the task. The failure probability at each step is `10%`.
```json
{
  "settings": {
    "name": "transport_robust_2",
    "render": true,
    "save": true,
    "failure_probability": 0.1
  },
  "agents": [
    {
      "id": "agent1",
      "cost": 1,
      "skills": [
        "transport"
      ]
    }
  ],
  "teams": [
    {
      "id": "team1",
      "k_robustness": 2,
      "agents": [
        "agent1",
        "agent1",
        "agent1",
        "agent1",
        "agent1",
        "agent1"
      ]
    }
  ],
  "goals": [
    {
      "id": "goal1",
      "tasks": [
        {
          "environment": "transport",
          "env_kwargs": {
            "n_packages": 1,
            "package_width": 0.2,
            "package_length": 0.2,
            "package_mass": 10
          }
        }
      ]
    }
  ]
}
```
