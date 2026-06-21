#include <Eigen/Dense>
#include <iostream>
#include "Network.h"
#include <fstream>

//helper function to read ints written on bigendian
int read_int_be(std::ifstream& file){
    unsigned char b[4];
    file.read(reinterpret_cast<char*>(b), 4);
    return (b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3];
}


int main(){
  Network net({4, 100, 2});

  std::cout << "Network created\n";

  std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> training_data;

  std::ifstream file("train-images-idx3-ubyte/train-images.idx3-ubyte", std::ios::binary);

  if(!file){
    std::cerr << "could not open file\n";
    return 1;
  }

  int magic = read_int_be(file);
  int num_img = read_int_be(file);
  int rows = read_int_be(file);
  int cols = read_int_be(file);
  std::cout << magic << " " << num_img << " " << rows << " " << cols << "\n";

  unsigned char pixels[784];
  file.read(reinterpret_cast<char*>(pixels), 784);

  for(int r = 0; r < rows; r++){
    for(int c = 0; c < cols; c++){
      if(pixels[(rows)*(r) + c] > 127){
        std::cout << "#";
      }else{
        std::cout << " ";
      }
    }
    std::cout << "\n";
  }

  unsigned char pixels1[784];
  file.read(reinterpret_cast<char*>(pixels1), 784);

  for(int r = 0; r < rows; r++){
    for(int c = 0; c < cols; c++){
      if(pixels1[(rows)*(r) + c] > 127){
        std::cout << "#";
      }else{
        std::cout << " ";
      }
    }
    std::cout << "\n";
  }






 



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

