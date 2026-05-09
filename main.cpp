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
  Network(const std::vector<int>& sizes){
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

  //this Sigmoid returns the activation vector 
  Eigen::VectorXd Sigmoid(const Eigen::VectorXd& z){
    Eigen::VectorXd result(z.size());

    for (int i = 0; i < z.size(); i++){
      double x = z(i);
      result(i) = 1.0/(1.0 + std::exp(-x));
    }
    
    return result;
  }

  //return the output of the network when given certain inputs
  Eigen::VectorXd FeedForward(Eigen::VectorXd a){
   
   for(int i = 0; i < num_layers-1; i++){
    a = Sigmoid(weights[i] * a + biases[i]);
   }

   return a;

  }

  



};