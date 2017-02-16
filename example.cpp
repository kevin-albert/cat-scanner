#include <iostream>
#include <random>
#include <cmath>

#include "lib/Eigen/Dense"
#include "cat.h"

using Vector = Eigen::VectorXf;
using Matrix = Eigen::MatrixXf;

// input layer
Vector x(256*256);

// hidden layer
Matrix Whx(100,256*256);
Vector bh(100);
Vector h(100);

// output layer
Matrix Wyh(2,100);
Vector by(2);
Vector y(2);

Vector y_expected(2);

// initialize everything
void init(void);
void sigmoid(Vector&);
float E();


int main(void) {
    init();

    // Forward Pass 1
    std::cout << "Forward pass 1\n";
    h = Whx * x + bh; 
    sigmoid(h);
    
    y = Wyh * h + by;
    sigmoid(y);

    std::cout << "Error: " << E() << "\n";
    
    // Backward Pass
    std::cout << "Backward pass\n";

    Vector delta_y = (y_expected-y).cwiseProduct(y).cwiseProduct(Vector::Constant(2,1) - y);
    Vector delta_h = (Wyh.transpose() * delta_y).cwiseProduct(h).cwiseProduct(Vector::Constant(100,1) - h);

    Matrix dWyh = delta_y * h.transpose();
    Matrix dWhx = delta_h * x.transpose();

    Vector dy = delta_y;
    Vector dh = delta_h;

    // Parameter Updates
    float mu = 0.1;
    
    Wyh += dWyh * mu;
    Whx += dWhx * mu;
    by += dy * mu;
    bh += dh * mu; 


    // Forward pass 2
    std::cout << "Forward pass 2\n";
    h = Whx * x + bh; 
    sigmoid(h);
    
    y = Wyh * h + by;
    sigmoid(y);

    std::cout << "Error: " << E() << "\n";
    return 0;
}


void init() {
    // convert cat to floats
    for (int i = 0; i < 256*256; ++i) {
        x[i] = ((float)cat[i])/255.0;
    }

    std::default_random_engine rng;

    // weights - uniform distro, mean of 1/sqrt(|fan-in|)
    std::normal_distribution<float> dist_h(0,1.0/sqrt(256));
    for (int i = 0; i < Whx.size(); ++i) Whx.data()[i] = dist_h(rng); 
    std::normal_distribution<float> dist_y(0,1.0/sqrt(100));
    for (int i = 0; i < Wyh.size(); ++i) Wyh.data()[i] = dist_y(rng); 

    // biases - uniform distro, stddev 1
    bh.setRandom();
    by.setRandom(); 

    // set expected 
    y_expected << 1, 0;
}


void sigmoid(Vector &v) {
    for (int i = 0; i < v.size(); ++i) {
        v[i] = 1.0 / (1 + std::exp(-v[i]));
    }
}


float E() {
    // sum of squares
    return (y_expected - y).squaredNorm();
}


