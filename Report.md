# Predicting Major League Baseball Pitch by Pitch Outcomes

## Introduction

The integration of data analytics in baseball throughout the past two decades has produced highly significant predictive models, especially in player development. This model aimed to predict the probabilities of 16 pitch outcomes based on various situational factors. This concept stemmed from a rationale that throughout an at-bat, the probabilities of the outcome are not uniform and can shift significantly based on factors such as pitch type, inning, outs, the change in balls and strikes, and the pitcher’s handedness. In essence, the probability of each pitch should not be independently evaluated but rather evaluated under the assumption that previous and real-time situational factors provide weight to the outcome. This model can help to predict said outcomes dependently, accounting for the underlying nuances of situational factors.

The data was derived from Kaggle and uses data from the 2019 Major League Baseball season. The data set consisted of 728,790 observations (at-bats) but only a random sample of half of this (364,395) was used for the model for faster processing time. 7 situational factors were used to predict the probabilities of the outcomes. The variables were batter, inning, pitch type, # of balls, # of strikes, # of outs, and pitcher handedness. There were 16 potential outcomes of each pitch as cited in Figure 1. A logistic regression model was used to estimate the probabilities of each outcome. 

![image](https://github.com/user-attachments/assets/10872194-26e9-4b90-8b3a-f06ddfc43ebd)


In the preliminary data analysis, the proportions of each outcome were calculated and displayed below.

![image](https://github.com/user-attachments/assets/d19dba7a-bcdb-4b43-94b1-8979beaa275e)


## Procedure

First, the necessary libraries needed for the model were loaded.

```python
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
```

The data was loaded into Python from an Excel file and variables of interest were analyzed.

```python
data = pd.read_excel("2019_pitches.xlsx")
print(data[['pitch_type', 'Batter', 'inning', 'b_count', 
  's_count', 'outs','on_1b','on_2b', 'on_3b', 'batter_hand', 
  'pitcher_throw', 'type']].head())
``` 

The data was then split in half to optimize the logistic regression processing time and a subset was created only examining relevant variables.

```python
from sklearn.model_selection import train_test_split
data_half1, data_half2 = train_test_split(data,   
  test_size=0.5, random_state=42)
selected_vars = ["Batter", "pitch_type", "inning", "b_count",
  "s_count", "outs", "on_1b", "on_2b", "on_3b", 
  "batter_hand", "pitcher_throw", "type"]
data_subset = data_half1[selected_vars]
```

The variable “type” that contained the outcomes of each pitch was converted into a categorical variable and dummy variables were created for each categorical variable.

```python
data_subset = data_half1[selected_vars].copy()
data_subset = pd.get_dummies(data_subset, columns=
  ["Batter","pitch_type", "inning", "batter_hand", 
  "pitcher_throw"], drop_first=True)
```

The independent and dependent variables were separated.

```python
X = data_subset.drop("type", axis=1)
y = data_subset["type"]
```

The logistic regression model was then created with L2 regularization.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

logreg_model = LogisticRegression(penalty='l2', max_iter=100)
logreg_model.fit(X, y)
```

With the logistic regression model developed, a new dataframe was created to set the parameters of the probability prediction of each outcome.

``` python
new_data = pd.DataFrame({
"pitch_type": [""],
"inning": [1], 
"b_count": [1], 
"s_count": [2], 
"outs": [1],
"on_1b": [0],
"on_2b": [0],
"on_3b": [0],
"batter_hand": [""],
"pitcher_throw": [""],
"Batter_Mookie Betts": [1], 
"pitch_type_CU": [1], 
"pitcher_throw_R": [1]
})
```

To change the parameters the following variables need to be adjusted:
1. Inning can be changed to the inning (1-9)
2. B_count can be changed to the number of balls in the count (0-3)
3. S_count can be changed to the number of strikes in the count (0-2)
4. Outs can be changed to the number of outs in the inning (0-2)
5. Batter_Player Name can be changed to the batter wanting to be analyzed (a 1 will need to be placed in the [] but the players name in the variable name should be changed – as seen in this example as “Batter_Mookie Betts”)
6. Pitch_type_pitch can be changed to the pitch type and similar to batter a 1 will remain in the [] with only the pitch symbol changed)

Dummy variables were retrieved for the categorical variables and missing columns were added.

```python
new_data = pd.get_dummies(new_data, columns=["pitch_type", 
  "batter_hand", "pitcher_throw"], drop_first=True)
missing_cols = set(X.columns) - set(new_data.columns)
for col in missing_cols:
  new_data[col] = 0
```

The probabilities were predicted for each class and the class names defined.

```python
predicted_probs = logreg_model.predict_proba(new_data)

class_names = {
0: "Ball in dirt",
1: "Ball",
2: "Called strike",
3: "In play, no out",
4: "In play, runs",
5: "Foul",
6: "Hit by pitch",
7: "Foul bunt",
8: "Missed bunt",
9: "Swinging pitchout",
10: "Pitchout",
11: "Swinging strike",
12: "Foul tip",
13: "Intentional ball",
14: "Swinging strike (Blocked)",
15: "In play, outs"
}
```

The predictions were sorted and displayed based on the set parameters.

```python
sorted_predictions = sorted(zip(class_names.values(), 
  predicted_probs.T), key=lambda x: x[1], reverse=True)

for class_num, (class_name, probs) in 
  enumerate(sorted_predictions, start=1):
  print(f"Class Name: {class_name} {probs}")
```

A plot was then created for easier visualization.

```python
sorted_class_names, sorted_probs = zip(*sorted_predictions)
for class_num, (class_name, probs) in 
  enumerate(sorted_predictions, start=1):
  print(f"Class Name: {class_name} {probs}")
plt.figure(figsize=(10, 6))
bars = plt.barh(sorted_class_names, sorted_probs, 
  color='blue')
plt.xlabel('Probability')
plt.ylabel('Outcome')
plt.title('Probabilities for Outcome')
plt.xlim(0, 1)
plt.gca().invert_yaxis()
for bar, prob in zip(bars, sorted_probs):
  plt.text(bar.get_width() + 0.01, bar.get_y() +  
  bar.get_height() / 2, f'{prob:.4f}', ha='left', 
  va='center')
plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/bc4c6534-6e47-4850-bef7-cc327871295a)


## Example of the Model

To test the model, we will use former Braves first basemen, Freddie Freeman, for this example at-bat sequence. He will face a right-handed pitcher in the 1st inning with 1 out and one baserunner on first base. These parameters will not change during this at bat but the count and pitch types will. The outcome with the highest probability for each pitch will be used to progress the count. 


Pitch 1

Batter: Freddie Freeman

Inning: 1

Pitcher Handedness: Right

Outs: 1

Balls: 0

Strikes: 0

Runners on First: 0

Runners on Second: 0

Runners on Third: 0

Predicted Pitch Type: Fastball

![image](https://github.com/user-attachments/assets/41f8d5f1-1202-4868-a492-8df3d22ef36d)
As seen a ball is the most likely outcome in this situation.

Pitch 2

Batter: Freddie Freeman

Inning: 1

Pitcher Handedness: Right

Outs: 1

Balls: 1

Strikes: 0

Runners on First: 0

Runners on Second: 0

Runners on Third: 0

Predicted Pitch Type: Curveball

![image](https://github.com/user-attachments/assets/e8130182-aacb-4a6c-a1e0-0751e811b306)
As seen another ball is the most likely outcome in this situation.

Pitch 3

Batter: Freddie Freeman

Inning: 1

Pitcher Handedness: Right

Outs: 1

Balls: 2

Strikes: 0

Runners on First: 0

Runners on Second: 0

Runners on Third: 0

Predicted Pitch Type: Fastball

![image](https://github.com/user-attachments/assets/0ccf39b4-7e23-4323-96d5-bdba9f9846f2)
As seen a foul ball is the most likely outcome in this situation.


From these three pitches, the outcome probabilities can be seen shifting as the count and predicted pitch type changes.

In conclusion, this model can serve an serve an impactful purpose as it can still be used as a baseline to understand batter trends in different situations. The most practical uses would be in player development to better prepare batters for common pitch types during specific counts, handedness of the pitcher, and number of outs. Additionally, this model could be useful for pitchers to understand which types of pitches to throw or not throw to certain hitters and in specific situations.
