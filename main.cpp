#include <Eigen/Dense>
#include <iostream>
#include "Network.h"
#include "Network2.h"
#include <fstream>

//helper function to read ints written on bigendian
int read_int_be(std::ifstream& file){
    unsigned char b[4];
    file.read(reinterpret_cast<char*>(b), 4);
    return (b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3];
}

//load training and test data
std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> load_data(std::string imagePath, std::string labelPath){
  std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> result;

  std::ifstream file(imagePath, std::ios::binary);
  std::ifstream fileLabels(labelPath, std::ios::binary);

  if(!file || !fileLabels){
    std::cerr << "could not open file \n";
  
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

    result.push_back({pixels, label});
  }
  std::cout << "all images and labels loaded";
  return result;

}



int main(){
  Network2 net({784, 40, 10});

  std::cout << "Network created\n";

  std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> training_data = load_data("train-images-idx3-ubyte/train-images.idx3-ubyte", "train-labels-idx1-ubyte/train-labels.idx1-ubyte");
  std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> test_data = load_data("t10k-images-idx3-ubyte/t10k-images.idx3-ubyte", "t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte");

  //train network
  net.SGD(training_data, 10, 30, 0.3, 0.001, 0.9);

  //evaluate
  int correct = 0;
  
  for(int i = 0; i < test_data.size(); i++){
    Eigen::VectorXd result = net.FeedForward(test_data[i].first);
    Eigen::Index predicted;
    Eigen::Index expected;
    result.maxCoeff(&predicted);
    test_data[i].second.maxCoeff(&expected);

    if(predicted == expected) correct++;  
  } 



  double accuracy = 100.0 * correct / test_data.size();

  std::cout << "neural accuracy: " << accuracy << "%" << "\n";

  net.SaveWeights("weights1.bin");

  return 0;
}

