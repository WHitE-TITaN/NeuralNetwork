#include "TextProcessing.h"
#include <sstream>



//loading the files take a signrfrcrnt amount of space
void userInput::loadGloveVectors(const std::string& gloveFile) {
	std::cout << "\n \n \t LOading Flies wait Might take a siginificant amount of time & space\n\n\t\t";
	std::ifstream file(gloveFile);
	if (!file.is_open()) {
		std::cerr << "\t !Could not open GloVe file!" << std::endl;
		return;
	}

	std::string line;
	while (std::getline(file, line)) {
		std::istringstream iss(line);
		std::string word;
		iss >> word;  // First token is the word

		std::vector<double> vector;
		double value;
		while (iss >> value) {
			vector.push_back(value);  // Remaining tokens are the vector values
		}

		glove_vectors[word] = vector;
	}

	std::cout << "Loaded GloVe vectors: " << glove_vectors.size() << std::endl;
	file.close();
}





// converting the sentence to tockens 
std::vector<std::string> userInput::tokenize(std::string& UInput) {
	std::vector<std::string> tockens;
	std::string tocken; 					
	std::stringstream word(UInput); // converting the string to stream
	while (word >> tocken){			// dividing words with incoming stream
		tockens.push_back(tocken);	// pushing the word to vector
		size++;
	}
	std::cout << "\n\n\tsize array ->> " << size;
	return tockens;
}


int userInput::Size() {
	return size;
}



// generating random vector for token
std::vector<double> userInput::RandomVectorGenerator() {
	std::vector<double> GenetreatedVector(size);  // vector array size of total no. of elements in text


	std::uniform_real_distribution<double> distribution(-1.0, 1.0);
	
	for (int i = 0; i < size; i++) {
		GenetreatedVector[i] = distribution(generator);
	}
	
	return GenetreatedVector;
}





//std::vector<std::vector<double>> userInput::convertTokensToVectors(const std::vector<std::string>& tokens) {
	//std::vector<std::vector<double>> vectors;

	//for (const auto& token : tokens) {
		//if (token_vectors.find(token) == token_vectors.end()) {
			// Generate a random vector for a new token
			//token_vectors[token] = RandomVectorGenerator();
		//}
	//	vectors.push_back(token_vectors[token]);
	//}

	//return vectors;
//}




/*std::vector<std::vector<double>> userInput::convertTokensToVectors(const std::vector<std::string>& tokens) {
	std::vector<std::vector<double>> IndependentVectors;
	for (auto token : tokens) {
		auto NewGeneratedVector = RandomVectorGenerator();
		IndependentVectors.push_back(NewGeneratedVector);
	}
	return IndependentVectors;
}*/



std::vector<std::vector<double>> userInput::convertTokensToVectors(const std::vector<std::string>& tokens) {
	std::vector<std::vector<double>> IndependentVectors;

	std::cout << "Sample GloVe vectors loaded:\n";
	for (const auto& pair : glove_vectors) {
		std::cout <<"  Vector size: " << pair.second.size() << "\n";
		break; // Only print the first word to avoid flooding the console
	}

	for (const auto& token : tokens) {
		if (glove_vectors.find(token) != glove_vectors.end()) {
			// Use the pre-trained GloVe vector
			IndependentVectors.push_back(glove_vectors[token]);
		}
		else {
			// Fall back to generating a random vector if word not found in GloVe
			auto NewGeneratedVector = RandomVectorGenerator();
			IndependentVectors.push_back(NewGeneratedVector);
		}
	}
	return IndependentVectors;
}



std::vector<double> userInput::averagePooling(const std::vector<std::vector<double>>& vectors) {
	std::vector<double> avg_vector(vectors[0].size(), 0.0);
	for (const auto& vec : vectors) {
		for (size_t i = 0; i < vec.size(); i++) {
			avg_vector[i] += vec[i];
		}
	}
	for (auto& val : avg_vector) {
		val /= vectors.size();
	}
	return avg_vector;
}