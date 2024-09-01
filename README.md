# Fake Job Posting Detection

Detecting fake job postings with deep learning

## Introduction

I have used deep learning to solve a binary classification problem: "Is this job description real? Isn't it a fake one?"

The used dataset can be found [here](https://www.kaggle.com/shivamb/real-or-fake-fake-jobposting-prediction).

## Method

After reading data from the CSV file they should be vectorized, so I used *tf-idf* algorithm for the strings. Then, I implemented a fully-connected neural network in *PyTorch* framework for processing those vectors:

```python
class Network(nn.Module):
	def __init__(self, input_size=NETWORK_INPUT_SIZE, output_size=NETWORK_OUTPUT_SIZE):
		super(Network, self).__init__()
		self.fc1 = nn.Linear(input_size, 256)
		self.fc2 = nn.Linear(256, 128)
		self.fc3 = nn.Linear(128, 64)
		self.fc4 = nn.Linear(64, 32)
		self.fc5 = nn.Linear(32, 16)
		self.fc6 = nn.Linear(16, 8)
		self.fc7 = nn.Linear(8, 4)
		self.fc8 = nn.Linear(4, output_size)

	def forward(self, x):
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)
		x = self.fc3(x)
		x = F.relu(x)
		x = self.fc4(x)
		x = F.relu(x)
		x = self.fc5(x)
		x = F.relu(x)
		x = self.fc6(x)
		x = F.relu(x)
		x = self.fc7(x)
		x = F.relu(x)
		x = self.fc8(x)
		return x
```

We have an imbalanced dataset for this binary classification problem. Because of that, I have used ```torch.nn.BCEWithLogitsLoss``` as my loss function. And for the cross-validation part, *skorch* library has been used in my code.

## Result

After running the code, a confusion matrix and some related statistics will be shown to you:

```
Predict     real        fake           
Actual
real        16864       150         
fake        384         482         


Overall Statistics: 

95% CI                                                            (0.96764,0.97263)
Kappa                                                             0.62834
NIR                                                               0.95157
Overall ACC                                                       0.97013

Class Statistics:

Classes                                                           real          fake             
ACC(Accuracy)                                                     0.97013       0.97013 
ERR(Error rate)                                                   0.02987       0.02987 
F0.5(F0.5 score)                                                  0.9804        0.71008 
F1(F1 score - harmonic mean of precision and sensitivity)         0.98441       0.64352 
F2(F2 score)                                                      0.98846       0.58838 
FN(False negative/miss/type 2 error)                              150           384     
FNR(Miss rate or false negative rate)                             0.00882       0.44342 
FP(False positive/type 1 error/false alarm)                       384           150     
FPR(Fall-out or false positive rate)                              0.44342       0.00882 
PPV(Precision or positive predictive value)                       0.97774       0.76266 
TN(True negative/correct rejection)                               482           16864   
TNR(Specificity or true negative rate)                            0.55658       0.99118 
TP(True positive/hit)                                             16864         482     
TPR(Sensitivity, recall, hit rate, or true positive rate)         0.99118       0.55658 
```

## Run

First of all, install the dependencies:

```bash
pip3 install -r requirements.txt
```

Then, run the project using Python version 3:

```bash
python3 main.py
```

