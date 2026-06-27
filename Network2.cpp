//this network2 has a softmax output layer and cross-entropy loss function
#include <Eigen/Dense>
#include <iostream>
#include "Network2.h"
#include <algorithm>
#include <random>
#include <fstream>


  //construct the class with random weights and biases
  Network2::Network2(const std::vector<int>& sizes){
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
  Eigen::VectorXd Network2::Sigmoid(const Eigen::VectorXd& z){
    Eigen::VectorXd result(z.size());

    for (int i = 0; i < z.size(); i++){
      double x = z(i);
      result(i) = 1.0/(1.0 + std::exp(-x));
    }
    
    return result;
  }

  //computes sigmoid prime 
  Eigen::VectorXd Network2::SigmoidPrime(const Eigen::VectorXd& z){
    Eigen::ArrayXd tempSigmoid = Sigmoid(z).array();
    return (tempSigmoid * (1 - tempSigmoid)).matrix();
  }

  //computes softmax output layer
  Eigen::VectorXd Network2::SoftMax(const Eigen::VectorXd& z){
    Eigen::VectorXd res = Eigen::VectorXd(z.size());
    int z_length = z.size();
    double a_divisor = 0;

    Eigen::Index m = z.maxCoeff();

    for(int j = 0; j < z_length; j ++){
       a_divisor+=std::exp(z[j] -m);
    }

    for(int i = 0; i < z_length; i++){

       double a_L_j = std::exp(z[i] - m) / a_divisor;
      
       res[i] = a_L_j;
    }
    
    return res;
  }


  //return the output of the network2 when given certain inputs
  Eigen::VectorXd Network2::FeedForward(Eigen::VectorXd a){
   
   for(int i = 0; i < num_layers-2; i++){
    a = Sigmoid(weights[i] * a + biases[i]);
   }

   a = SoftMax(weights[num_layers - 2] * a + biases[num_layers -2]);
  

   return a;

  }

  //backpropagation algorithm, computes gradients for weights and biases. returns nabla_b nabla_w
  std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::MatrixXd>> Network2::Backprop(const Eigen::VectorXd& x, const Eigen::VectorXd& y){
   
   //store weighted sums and activations at each layer during a forward pass.
   std::vector<Eigen::VectorXd> temp_z;
   std::vector<Eigen::VectorXd> temp_a;


   //store gradients
   std::vector<Eigen::VectorXd> nabla_b;
   std::vector<Eigen::MatrixXd> nabla_w;

   //temp activations and Z
   Eigen::VectorXd a(x);
   
   //push the first activation layer (input)
   temp_a.push_back(a);
   Eigen::VectorXd z;

   //initialize gradients to zeros
   for(int i = 0; i < num_layers-1; i++){    
     nabla_b.push_back(Eigen::VectorXd::Zero(biases[i].size()));
     nabla_w.push_back(Eigen::MatrixXd::Zero(weights[i].rows(), weights[i].cols()));
   }

   //**forward pass** 
   for(int i = 0; i < num_layers-2; i++){
     //compute Z vector for each layer, append to the list of Z, the same for activations
     z = weights[i] * a + biases[i];
     temp_z.push_back(z);
     a = Sigmoid(z);
     temp_a.push_back(a);
   }

   //softmax output layer
   z = weights[num_layers - 2] * a + biases[num_layers - 2];
   temp_z.push_back(z);
   a = SoftMax(z);
   temp_a.push_back(a);

   //**Backward pass **/
   
   //compute outtermost error vector first (cross entropy loss function)
   Eigen::VectorXd delta_a = temp_a.back() - y; 
   nabla_b.back() = delta_a;
   nabla_w.back() = delta_a * temp_a[num_layers-2].transpose();
 
   for(int l = num_layers - 3; l >= 0; l--){
     delta_a = ((weights[l+1].transpose() * delta_a).array() * SigmoidPrime(temp_z[l]).array()).matrix();
     nabla_b[l] = delta_a;
     nabla_w[l] = delta_a * temp_a[l].transpose();    
   }
   
   return {nabla_b, nabla_w};

  }

  //update weights and biases based on gradients and learning rate
  void Network2::UpdateMiniBatch(const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& mini_batch, double eta, double decayRate, int training_size){   
   
    //store average gradients
   std::vector<Eigen::VectorXd> average_nabla_b;
   std::vector<Eigen::MatrixXd> average_nabla_w;

   //initialize gradients to zeros
   for(int i = 0; i < num_layers-1; i++){    
     average_nabla_b.push_back(Eigen::VectorXd::Zero(biases[i].size()));
     average_nabla_w.push_back(Eigen::MatrixXd::Zero(weights[i].rows(), weights[i].cols()));
   }

   //sum up gradients
   for(int i = 0 ; i < mini_batch.size(); i++){
    auto gradients = Backprop(mini_batch[i].first, mini_batch[i].second);
    
    //aggregate gradients
    for(int l = 0; l < num_layers-1; l++){    
     average_nabla_b[l] += gradients.first[l];
     average_nabla_w[l] += gradients.second[l];
    }
   }

   //update weights
   for(int i = 0; i < num_layers-1; i++){    
     biases[i] -= (average_nabla_b[i] / mini_batch.size()) * eta;
     weights[i] *= (1 - ((eta * decayRate)/training_size)); //weight decay
     weights[i] -= (average_nabla_w[i] / mini_batch.size()) * eta;
   }
  }

  //update net weights based training data
  void Network2::SGD(std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& training_data, int mini_batch_size, int epochs, double eta, double decayRate){

    std::random_device rd;
    std::mt19937 gen(rd());

    int training_size = static_cast<int>(training_data.size());

    if(mini_batch_size >= training_size) mini_batch_size = training_size;
    if (training_size == 0) return;


    //train
    for(int i = 0; i < epochs; i++){ //outter loop trains foreach each epoch
      std::cout << "Epoch " << i << " complete\n";
      //shuffle training data
      std::shuffle(training_data.begin(), training_data.end(), gen);

      for(int j = 0; j < training_size; j+=mini_batch_size){//inner loop creates minibatch

       int lastIndex = j + mini_batch_size;
       if(lastIndex > training_size) lastIndex = training_size;

       std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> mini_batch;
       mini_batch.reserve(lastIndex-j);

       for(int h = j; h < lastIndex; h++){
        mini_batch.push_back(training_data[h]);
       }

       //minibatch created. update weights
       UpdateMiniBatch(mini_batch, eta, decayRate, training_size);

      }

    }

  }

  //saves to <filename> netSize, sizes (length is netSize), weights as 8byte doubles, and finally biases.  
  void Network2::SaveWeights(std::string filename){
   std::ofstream file(filename, std::ios::binary);
   file.write(reinterpret_cast<char*>(&num_layers), sizeof(int));
   file.write(reinterpret_cast<const char*>(sizes.data()), (num_layers * sizeof(int)));
   

   for(int i = 0; i < num_layers - 1; i++){
      file.write(reinterpret_cast<const char*>(weights[i].data()), weights[i].size() * sizeof(double));
   }
   
   for(int i = 0; i < num_layers - 1; i++){
      file.write(reinterpret_cast<const char*>(biases[i].data()), biases[i].size() * sizeof(double));
   }
   
   
  }
    
  

  

