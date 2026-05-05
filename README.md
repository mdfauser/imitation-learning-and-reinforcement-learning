# Imitation and Reinforcement Learning exercise for Deep Learning Lab 2026. 
Tested with python 3.11

Recommended virtual environments: `conda` or `virtualenv`.

Activate your virtual environment and install dependencies. Please don't skip the installation of pip, setuptools and wheel with the version we specified.
```[bash]
pip install -r requirements.txt
```

Please format your code with `black .` before submission.

## Imitation Learning
Data Collection
```[bash]
python imitation_learning/drive_manually.py
```

Training
```[bash]
python imitation_learning/training.py
```

Testing
```[bash]
python imitation_learning/test.py
```

## Reinforcement Learning

RL Agents Learning
```[bash]
python reinforcement_learning/train_carracing.py
```

```[bash]
python reinforcement_learning/train_cartpole.py
```

RL Agents Testing
```[bash]
python reinforcement_learning/test_carracing.py
```

```[bash]
python reinforcement_learning/test_cartpole.py
```