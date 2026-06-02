#include <Eigen/Dense>
#include <iostream>
#include "Network.h"


  //construct the class with random weights and biases
  Network::Network(const std::vector<int>& sizes){
    this->num_layers = sizes.size();
    this->sizes = sizes;

    for(int i = 1; i < num_layers; i++){
     //fill the biases with random values first 
     Eigen::VectorXd tempBias = Eigen::VectorXd::Random(sizes[i]);
     biases.push_back(tempBias);

     //fill the weights with random values
     Eigen::MatrixXd tempWeights = Eigen::MatrixXd::Random(sizes[i], sizes[i-1]);
     weights.push_back(tempWeights);
      
    }
  }

  //this Sigmoid returns the activation vector, takes as input the weighted sum vector 
  Eigen::VectorXd Network::Sigmoid(const Eigen::VectorXd& z){
    Eigen::VectorXd result(z.size());

    for (int i = 0; i < z.size(); i++){
      double x = z(i);
      result(i) = 1.0/(1.0 + std::exp(-x));
    }
    
    return result;
  }

  //return the output of the network when given certain inputs
  Eigen::VectorXd Network::FeedForward(Eigen::VectorXd a){
   
   for(int i = 0; i < num_layers-1; i++){
    a = Sigmoid(weights[i] * a + biases[i]);
   }

   return a;

  }

  //computes sigmoid prime 
  Eigen::VectorXd Network::SigmoidPrime(const Eigen::VectorXd& z){
    Eigen::ArrayXd tempSigmoid = Sigmoid(z).array();
    return (tempSigmoid * (1 - tempSigmoid)).matrix();
  }


