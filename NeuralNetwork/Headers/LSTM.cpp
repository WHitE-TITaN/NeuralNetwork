#include "LSTM.h"

double sigmoid_Activation(double entity_Vector) {
	//will map any x axis coordinate to its corosponding y axis coordiante
	//betwqeen 0-1;
	return exp(entity_Vector) / exp(entity_Vector) + 1;
}

double tanh_Activation(double entity_Vector) {
	//will map any x axis coordinate to its corosponding y axis coordiante
	//betwqeen 1 & -1;
	return (exp(entity_Vector) - exp(-entity_Vector)) / (exp(entity_Vector) + exp(-entity_Vector));
}



void forgetGate(double currentInputVector) {
	double activation = (shortTermMemory * longTermRemember.shortTermWeight)
		+ (currentInputVector * longTermRemember.inputWeight) + longTermRemember.bias; //adding up the equations

	longTermMemory = longTermMemory * sigmoid_Activation(activation); // adding to long term memory
}


void inputGate(double currentInputVector) {
	//to calculate potential to add up to long term memory
	double activation = (shortTermMemory * potentialLTM.shortTermWeight)
		+ (currentInputVector * potentialLTM.inputWeight) + potentialLTM.bias; // adding up the values

	double potentialActivation = tanh_Activation(activation);

	//to calculate the ammount of potential to be added
	double activation = (shortTermMemory * percentageLTMPotential.shortTermWeight)
		+ (currentInputVector * percentageLTMPotential.inputWeight) + percentageLTMPotential.bias;
	
	double percentagePotential = sigmoid_Activation(activation);

	longTermMemory = longTermMemory + (percentagePotential * potentialActivation);
}


void outputGate(double currentInputVector) {
	//calculating the potential short term memory 
	double potentialSTM = tanh_Activation(longTermMemory);

	double activation = (shortTermMemory * percentageSTMPotental.shortTermWeight)
		+ (currentInputVector * percentageSTMPotental.inputWeight) + percentageSTMPotental.bias; // adding up the values
	//calculate actual percentage to account for
	double percentagePotential = sigmoid_Activation(activation);

	shortTermMemory = percentagePotential * potentialSTM; // update short term memory


}