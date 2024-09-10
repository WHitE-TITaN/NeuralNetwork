// NeuralNetwork.cpp : Defines the entry point for the application.

#include <iostream>
#include "Headers/TextProcessing.h"
#include "Headers/SelfAttention.h"
#include <string>
#include <vector>




int main() {
	userInput dk488621;

	//loading the files
	dk488621.loadGloveVectors("C:/Users/dk488/source/repos/vocabulary/glove.txt");

	std::cout << "\n\n\n\t\t\t\tWELLCOME";
	std::string UInput;


	while (true) {
		std::cout << "\n\t\t May i help you ->> ";
		std::getline(std::cin, UInput); //taking the input ->> user 
		if (UInput == "exit" || UInput == "quit" || UInput == "00") {
			break;
		}

		std::vector<std::string> processedString = dk488621.tokenize(UInput); // the tocknized version of the input 

		std::cout << "\n\t\t TOKENIZED -> "; // printing the tockens 
		for (auto T : processedString) {
			std::cout << "\n\t\t            " << T;
		}



		//generating random vectors for every word ->
		std::vector<std::vector<double>> VectorGenerated = dk488621.convertTokensToVectors(processedString);
		std::cout << "\n\n {   ";
		for (auto x : VectorGenerated) {
			std::cout << "\t\t<";
			for (auto y : x) {
				std::cout << "[ " << y << " ], ";
			}
			std::cout << ">\n";
		}
		std::cout << "   } \n\n";
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



	/*std::vector<std::vector<double>> InterDependentVector = DynamicMultiplication(VectorGenerated, dk488621.size);
	std::cout << "\n\t\t\tCONTEXT AWAIR MATRIX DYNAMIC PROGRAMMING\n\n\n {   ";
	for (auto x : InterDependentVector) {
		std::cout << "\t\t<";
		for (auto y : x) {
			std::cout << "[ " << y << " ], ";
		}
		std::cout << ">\n";
	}
	std::cout << "   } \n\n";




	std::vector<std::vector<double>> InterDependentVector2 = Softmax(InterDependentVector, dk488621.size);
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
