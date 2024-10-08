#ifndef TextProcessing_H
#define TextProcessing_H

#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <unordered_map>
#include "SelfAttention.h"




class userInput {
	public:
		std::vector<std::string> tokenize(std::string& UInput);
		std::vector<std::vector<double>> convertTokensToVectors(const std::vector<std::string>& tokens);
		int size = 0;

		//the loaded vector -> 
		std::unordered_map<std::string, std::vector<double>> glove_vectors;

		//loading the pretrainned data to the memory for the execution
		void loadGloveVectors(const std::string& gloveFile);
		std::vector<double> averagePooling(const std::vector<std::vector<double>>& vectors);

	private:
		std::vector<double> RandomVectorGenerator();
		std::default_random_engine generator;
		std::unordered_map<std::string, std::vector<double>> token_vectors;
		int Size();
	
};


#endif