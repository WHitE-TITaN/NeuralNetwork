#ifndef LSTM_H
#define LSTM_H

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <unordered_map>

double longTermMemory=0 /* cell state */,
	shortTermMemory = 0; /* Hidden state */

class layer {
	public:
		double shortTermWeight, /* weight of the short t6erm memory*/
			inputWeight, /* weight of the input */
			bias; /* over all percentage to be added buy the layer */

		layer() {
			shortTermWeight = randomWeight();
			inputWeight = randomWeight();
			bias = randomBias();
		}

	private:
		double randomWeight() {
			// Initialize with small random weights
			return (double)rand() / RAND_MAX - 0.5;
		}

		double randomBias() {
			return (double)rand() / RAND_MAX - 0.5;
		}
};

//declaring layers to store the wights and biases 
layer longTermRemember;
layer percentageLTMPotential, potentialLTM, percentageSTMPotental;


double sigmoid_Activation(double entity_Vector);
double tanh_Activation(double entity_Vector);


void forgetGate(double currentInputVector); /* determine how much the longterm memory will be affected both + & - */
void inputGate(double currentInputVector); /* determines how the long term memory willl be updated */
void outputGate(double currentInputVector); /* determines the out put or future and the short term memory update */
std::vector<double> lstmMainFlow(std::vector<double> wordEmbedding);

#endif