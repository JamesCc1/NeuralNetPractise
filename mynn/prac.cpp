#include <iostream>
#include <vector>
#include <memory>
#include "tensor.h"
#include "Matrix.h"
#include "utils.h"
#include "batch.h"
#include "Sequential.h"
#include "Operator.h"
#include "softmax.h"
#include "linearOp.h"
#include "Relu.h"
#include "CrossEntropy.h"
#include "Sgd.h"
#include "read.h"


using namespace std;
using namespace Mini;

int main(){

    nn::Sequential model;
    cout << "Linear 1." << endl;
    model.add_block(nn::Linear::Construct(784, 256)); 
    cout << "Linear 2." << endl;
    model.add_block(nn::Linear::Construct(256, 10));
    cout << "Softmax." << endl;
    model.add_block(nn::Softmax::Construct());
    cout << "Loss Layer." << endl;
    model.set_loss(nn::CrossEntropyLoss::Construct());
    model.use(nn::SGD::Construct(0.004, 0.5));


    cout << "Load Data Set" << endl;
    char path[] = "data/trainset.txt";
    MNIST_LOAD dataloader(path);
    dataloader.load_data();
   
    int total = 150;
    int batch_size = 64;
    for(int epoch=1; epoch<=total; epoch++){
        cout << epoch << ':' << " Running." << endl;
        while(!dataloader.atEnd()){
            //load data
            auto x = dataloader.batch_load(batch_size);    
            auto data = x.first;
            auto labels = x.second; 

            // train
            auto res = model(data);
            auto loss = model.compute_loss(res, labels);
            model.back_step();
            model.zero_grad();

            F loss_avg = 0.0;
            for(int k=0; k<loss.batch_num(); k++) {
                loss_avg += loss[k].at(0);
            }
            cout << loss_avg / loss.batch_num() << endl;
        }
    }
    
    cout << "Test End." << endl;
    

}
