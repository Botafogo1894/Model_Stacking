# Model_Stacking
Using Model-Stacking and Deep Learning NNs to boost model performance and accuracy of my NLP models in Project3.

## What is Model Stacking?

Model stacking is a Data Science ensemble method, which relies on the "wisdom of the crowd" premise that a diverse selection of combined "weaker" learners working together, will often outperform a single "strong" model. The winners of most major Kaggle competitions over the past 4–5 years have used some configuration of Model Stacking in their final winning models.
Model Stacking is analogous with real-world examples such as building successful human teams in business, science, sports etc. If all the members of the team were really good at the exact same task, then the team would crush any challenge that requires this one specific skill, but it would fail miserably when it comes to handling complex real-life problems that require a plethora of diverse skills, mindsets, and approaches. I do not know much about American football, but even with my limited knowledge, it is pretty obvious to me that you cannot win a football game with a team that consists of only quarterbacks, even if those quarterbacks are the best in the league. That is why optimal sports teams and successful business units consist of a diverse group of individuals with a wider range of strengths and weaknesses.
## How does Model Stacking work in practice?

Below you can see an example of the top winning model of a recent major Kaggle competition:

![](https://github.com/Botafogo1894/Model_Stacking/blob/master/Keggle%2064%20models.png)

__Note:__ It is very important to have a sufficient amount of data in order to perform robust Model Stacking. To avoid over-fitting, you need to perform cross-validation at each stacking/training stage and keep some data aside as a "holdout" set for the testing stage and make sure that there isn't a huge discrepancy between the model's performance on train and test data.

 - **Initial Stage** - you run a variety of different standalone models and spend some time analyzing their individual performance metrics and thinking about where some models might have done better than others.
 - **Stage 1 Ensemble** - you select a small "team" of those models, making sure there is a low correlation between their prediction coefficients to ensure that your Stacked Model allows for a lot of cross-learning between the weaker links. You take the average of their predictions and construct a new table, which will be fed into a new smaller team of models in 
 - **Stage 2 Ensemble** - you run a new ensemble of models, which will use the averaged prediction metrics from Stage 1 as features in order to learn new information about the relationships between our original variables.
- **Stage 3 Ensemble** - you repeat the same process, feeding the average predictions of the models from Stage 2 into a final "meta-learner" model, which should be well selected to fit the type of problem that you're trying to solve. In the example above, the competitors used Linear Regression for their final model because this was probably one of the models that performed best during the initial stage as a standalone model.

## My process for Model Stacking
For my first experiment with Model Stacking, I decided to expand on my latest Data Science project, which was a Natural Language Processing model that aims to predict a song's genre from its lyrics. For that project, we preprocessed a list of 16,000 song lyrics from eight different genres - Hip-Hop, Country, Pop, Rock, Metal, Electronic, Jazz, and R&B. We made sure to include 2000 songs per genre in our dataset to avoid the issue of class imbalance. The first step was to run a diverse set of basic models with a low correlation between their predictive methods in order to be able to build our Model Stacking teams accordingly. Please see results below:

![](https://github.com/Botafogo1894/Model_Stacking/blob/master/basic_5_models.png)

Next, we decided to take the top 3 models and do some hyperparameter optimization via an extensive GridSearch, in order to get a sense for the highest accuracy that our top standalone model can achieve. Results below:

![](https://github.com/Botafogo1894/Model_Stacking/blob/master/top_3_models.png)

As you can see from the graph below, our top performer achieved a **testing accuracy of 50%**, which at first sight doesn't seem very high, but given that our model is trying to predict what genre a song belongs to out of 8 possible genres, this accuracy is **exactly four times better than random guessing** (probability of a random guess = 1/8 or 12.5%). This means that about half the time our model predicts exactly what genre a song belongs to, only based on that song's lyrics and that 50% testing accuracy was achieved on a testing set of over 3000 songs…not bad at all!

## But can we do better using Model Stacking and Neural Netwroks? 

To answer this question, first, I opted to combine and average the predictions of my three weakest learners - Random Forest, AdaBoost, and KNN classifiers and construct a new dataframe of features that I can feed into my strong learners. Code below:

![](https://github.com/Botafogo1894/Model_Stacking/blob/master/Round%201%20Stacking.png)

Each column in the table above is the average prediction coefficient for each of the eight genres and each row is a song lyric with a corresponding true genre, stored in a separate y-variable as a list of class labels.

Next, I split the data in a train and test set and ran Stage 2 of the Model Stacking, where my strong learners - GradientBoost and Naive Bayes - used the combined predictions from the weak learners to generate a new set of prediction. Then, I combined these predictions and used them as features into my final meta-learner - NN. Code below:

![](https://github.com/Botafogo1894/Model_Stacking/blob/master/Final%20NN%20model.png)

I opted to go with a Neural Network for the final stage of my Model Stacking because NNs tend to perform very well on multi-classification problems and NNs are pretty good at finding hidden links and figuring out complex relationships between dependent and independent variables. I used "softmax" for my output layer activation function because we're trying to predict eight classes. Lo and behold, results below…

![](https://github.com/Botafogo1894/Model_Stacking/blob/master/final%20perf.png)

**The 80% accuracy** was generated on a "holdout" set of test data, which consisted of 3600 song lyrics that the NN model had never seen before. The training accuracy of 86% with cross-validation was also very impressive, and what made me especially happy was the fact that there wasn't a significant difference between the training set and the testing set performance, so **there didn't seem to be much overfitting.**
