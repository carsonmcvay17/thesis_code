# Undergrad Thesis Code
The goal of this project was to determine if artificial neural networks trained on MNIST digits are poised at criticality. 
This code has 3 main functions:\
-Create a neural network designed to classify MNIST digits\
-Plot the heat capacity of the activations of the neural network and see if it matches the heat capacity plots of known critical systems\
-Create energy vs entropy plots to further confirm criticality

Disclaimer: There might be some bugs in the code, I have quickly made a repo of the code
because looking at my old google colab doc was making me upset at how bad the code was

# Organization
This code is organized into a few folders\
-`figures`: figures stores all the nice and fun plots\
-`sources`: The bulk of the code is included in the sources. Sources includes code to calculate heat capacity, entropy, energy, etc.\
-`scripts`: Runs the code written in sources, makes the pretty plots

# Dependencies
-tensorflow\
-matplotlib\
-keras\
-numpy\
-math\
-scipy\