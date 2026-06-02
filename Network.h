#pragma once
#include <Eigen/Dense>
#include <iostream>

class Network{
 private :
   int num_layers;
   std::vector<int> sizes;
   std::vector<Eigen::VectorXd> biases;
   std::vector<Eigen::MatrixXd> weights;

 
  public:
  //initialize the network with random weights and biases 
  Network(const std::vector<int>& sizes);

  //this Sigmoid returns the activation vector 
  Eigen::VectorXd Sigmoid(const Eigen::VectorXd& z);
  //return the output of the network when given certain inputs
  Eigen::VectorXd FeedForward(Eigen::VectorXd a);
  //returns the derivative of the sigmoid function
  Eigen::VectorXd SigmoidPrime(const Eigen::VectorXd& z);

  



};