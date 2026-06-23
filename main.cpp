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
  Network net({784, 30, 10});

  std::cout << "Network created\n";

  std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> training_data;

  std::ifstream file("train-images-idx3-ubyte/train-images.idx3-ubyte", std::ios::binary);
  std::ifstream fileLabels("train-labels-idx1-ubyte/train-labels.idx1-ubyte", std::ios::binary);
  if(!file || !fileLabels){
    std::cerr << "could not open file \n";
    return 1;
  }

  int magic = read_int_be(file);
  int num_img = read_int_be(file);
  int rows = read_int_be(file);
  int cols = read_int_be(file);

  int magicLabel = read_int_be(fileLabels);
  int numLabels = read_int_be(fileLabels);


  std::cout << magic << " " << num_img << " " << rows << " " << cols << " " << "magicNumLabels:" << magicLabel << " numLabels:" << numLabels << "\n";

  //save all images and labels
  for(int i=0; i < num_img; i++){
    //read pixel data
    Eigen::VectorXd pixels(784);
    unsigned char pixelsBytes[784];
    file.read(reinterpret_cast<char*>(pixelsBytes), 784);
 
    //read label data
    Eigen::VectorXd label = Eigen::VectorXd::Zero(10);
    unsigned char labelByte;
    fileLabels.read(reinterpret_cast<char*>(&labelByte), 1);
    label[labelByte] = 1;

    //normalize pixel data
    for(int j = 0; j < 784; j++){
      pixels[j] = pixelsBytes[j] / 255.0;
    }

    training_data.push_back({pixels, label});
  }
  std::cout << "all images and labels loaded";


  //train network
  net.SGD(training_data, 10, 30, 3);

  double net_loss = 0.0;
  
  for(int i = 0; i < training_data.size(); i++){
    Eigen::VectorXd result = net.FeedForward(training_data[i].first);
    net_loss += (result - training_data[i].second).squaredNorm();
  } 

  net_loss /= training_data.size();

  std::cout << "neural net loss: " << net_loss << "\n";

  return 0;
}

