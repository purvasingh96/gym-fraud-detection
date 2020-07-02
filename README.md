# gym-fraud-detection

* [How to create new gym environment in openai](https://github.com/openai/gym/blob/master/docs/creating-environments.md)

# How to create new environments for Gym

* Create a new repo called gym-fraud-detection, which should also be a PIP package.

* A good example is https://github.com/openai/gym-soccer.

* It should have at least the following files:
  ```sh
  gym-fraud-detection/
    README.md
    setup.py
    gym_fraud_detection/
      __init__.py
      envs/
        __init__.py
        fraud_detection_env.py
  ```

* `gym-foo/setup.py` should have:

  ```python
  from setuptools import setup

  setup(name='gym_fraud_detection',
        version='0.0.1',
        install_requires=['gym'] 
  )
  ```

* `gym-fraud-detection/gym_fraud_detection/__init__.py` should have:
  ```python
  from gym.envs.registration import register

  register(
      id='gym-fraud-detection-v0',
      entry_point='gym_fraud_detection.envs:FraudDetectionEnv',
  )
  ```

* `gym-fraud-detection/gym_fraud_detection/envs/__init__.py` should have:
  ```python
  from gym_fraud_detection.envs.fraud_detection_env \
  import FraudDetectionEnv
  ```

* `gym-fraud-detection/gym_fraud_detection/envs/fraud_detection_env.py` should look something like:
  ```python
  import gym
  from gym import error, spaces, utils
  from gym.utils import seeding

  class FraudDetectionEnv(gym.Env):
    def __init__(self):
      ...
    def step(self, action):
      ...
    def reset(self):
      ...
    def render(self, mode='human'):
      ...
    def close(self):
      ...
  ```

* After you have installed your package with `pip install -e gym-foo`, you can create an instance of the environment with `gym.make('gym_foo:foo-v0')`

# Installation 
```
cd gym-fraud-detection
pip install -e .
```

# Usage 

**Step - 1 :** Create a directory named *dataset* in your folder containing the main program.<br>
**Step - 2 :** Download [Kaggle's Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place it inside *dataset folder*<br>
**Step - 3 :** In your code create an instance of gym_fraud environment using the following commands <br>
 ```python
import gym
import gym_fraud_detection
import FraudDetectorAgent
env = gym.make('fraud-detection-v0')

obs = env.reset()
agent = FraudDetectorAgent.Agent()
while True:
 action = agent.epsilon_greedy_action(obs)
 next_state, reward, done, info = env.step(action)
if done:
  break
```
