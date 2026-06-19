In this mini-project, I'm trying to code a binary classifier from scratch, using only numpy.

To execute code: python main.py
Results are printed in the terminal + images in images/


Structure

project/
    main.py
    src/
    images/
    README.md
    requirements.txt
    .gitignore


Training data: points in the plane X
- 200 randomly distributed with gaussian distribution around (-2, -2), with label y=0
- 200 randomly distributed with gaussian distribution around (2, 2), with label y=1


1st approach: logistic regression

Initial parameters: w = (0, 0) and b = 0
Initial accuracy: 0.5
Score : z = w.x + b
Prediction: yhat = sigmoid(z)

Loss function used: binary cross-entropy, the mean value of -y * log(yhat) - (1 - y) * log(1 - yhat)


Inference on new points:
200 points following a 2D Gaussian distribution. 
Their predicted class can be seen in prediction.png