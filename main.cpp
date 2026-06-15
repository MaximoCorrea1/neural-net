#include <Eigen/Dense>
#include <iostream>
#include "Network.h"

int main(){
  Network net({2, 100, 2});

  std::cout << "Network created\n";

  std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> training_data;

  Eigen::VectorXd x1(4);
  x1 << 0.0, 0.0, 1.0, 1.0;

  Eigen::VectorXd y1(2);
  y1 << 1.0, 1.0;

  training_data.push_back({x1, y1});

  Eigen::VectorXd x2(4);
  x2 << 1.0, 1.0, 0.0, 0.0;

  Eigen::VectorXd y2(2);
  y2 << 0.0, 1.0;

  training_data.push_back({x2, y2});


  //train network
  net.SGD(training_data, 1, 1000, 0.2);

  double net_loss = 0.0;
  
  for(int i = 0; i < training_data.size(); i++){
    Eigen::VectorXd result = net.FeedForward(training_data[i].first);
    net_loss += (result - training_data[i].second).squaredNorm();
  } 

  net_loss /= training_data.size();

  std::cout << "neural net loss: " << net_loss << "\n";

  return 0;
}

