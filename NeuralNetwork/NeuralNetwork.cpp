﻿// NeuralNetwork.cpp : Defines the entry point for the application.

#include <iostream>
#include "Headers/TextProcessing.h"
#include "Headers/SelfAttention.h"
#include "Headers/LSTM.h"
#include <string>
#include <vector>




int main() {
	userInput dk488621;

	//loading the files
	dk488621.loadGloveVectors("C:/Users/dk488/source/repos/vocabulary/glove.txt");
	bool backPropogation = false;
	std::cout << "\n\n\n\t\t\t\tWELLCOME";
	std::string UInput;
	std::vector<std::string> backProcessedString;
	std::vector<double> reducedBackPropVector;
	dk488621.addEOS(dk488621.glove_vectors, 50);

	while (true) {
		std::cout << "\n\t\t May i help you ->> ";
		std::getline(std::cin, UInput); //taking the input ->> user 



		if (UInput == "exit" || UInput == "quit" || UInput == "00") {
			break;
		}
		if (UInput == "/mode1") {
			backPropogation = true;
			std::cout << "\n mode enabled ->>";
			continue;
		}
		else if (UInput == "/mode0") {
			backPropogation = false;
			std::cout << "\n mode desable";
			continue;
		}

		std::vector<std::string> processedString = dk488621.tokenize(UInput); // the tocknized version of the input 

		std::cout << "\n\t\t TOKENIZED -> "; // printing the tockens 
		for (auto T : processedString) {
			std::cout << "\n\t\t            " << T;
		}



		//generating random vectors for every word ->
		std::vector<std::vector<double>> VectorGenerated = dk488621.convertTokensToVectors(processedString);

		std::vector<double> ReducedDimensionality = dk488621.averagePooling(VectorGenerated);
		std::cout << "\n\n AVERAGE POOLED \n {   ";
		for (auto x : ReducedDimensionality) {
			std::cout <<"> " << x<<"   ";
		}
		std::cout << "   } \n\n";


		if (backPropogation == true) {
			std::cout << "Enter the actual value ->> ";
			std::getline(std::cin, UInput);
			backProcessedString = dk488621.tokenize(UInput); // the tocknized version of the input 
			std::vector<std::vector<double>> backVectorGenerated = dk488621.convertTokensToVectors(backProcessedString);
			reducedBackPropVector = dk488621.averagePooling(backVectorGenerated);


			// back propgation ->>

			std::cout << "\n THE ACTUAL OUTPUT \n\n {   ";
			for (auto x : reducedBackPropVector) {
				std::cout << ">" << x << " ";
			}
			std::cout << "   } \n\n";


			std::vector<double>finalOutPut = lstmMainFlow(ReducedDimensionality);

			lstmBackprop(finalOutPut, reducedBackPropVector, ReducedDimensionality);
			std::cout << "\n THE OUTPUT \n\n {   ";
			for (auto x : finalOutPut) {
				std::cout << ">" << x << " ";
			}
			std::cout << "   } \n\n";

			//std::vector<std::string> mappedWords = mapOutputToWords(dk488621.glove_vectors, finalOutPut, 3); // Top 3 words
			//std::string constructedPhrase = assemblePhrase(mappedWords);

			std::string generatedSentence = generateSentence(dk488621.glove_vectors, finalOutPut);

			std::cout << "Constructed Phrase: " << generatedSentence << std::endl;

		}



	}




	//Developing depencency and context awair 
	/*std::vector<std::vector<double>> InterDependentVector = ContextAwair(VectorGenerated, dk488621.size);
	std::cout << "\n\t\t\tCONTEXT AWAIR MATRIX\n\n\n {   ";
	for (auto x : InterDependentVector) {
		std::cout << "\t\t<";
		for (auto y : x) {
			std::cout << "[ " << y << " ], ";
		}
		std::cout << ">\n";
	}
	std::cout << "   } \n\n";*/



	/*std::vector<std::vector<double>> InterDependentVector2 = Softmax(InterDependentVector, dk488621.size);
	std::cout << "\n\t\t\t\t\tSOFTMAX NORMILIZATION \n\n\n {   ";
	for (auto x : InterDependentVector2) {
		std::cout << "\t\t<";
		for (auto y : x) {
			std::cout << "[ " << y << " ], ";
		}
		std::cout << ">\n";
	}
	std::cout << "   } \n\n";*/




	return 0;

}
