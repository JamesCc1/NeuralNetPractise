#include <iostream>
#include <vector>

#include "shape.h"
#include "tensor.h"
#include "Matrix.h"
#include "linearOp.h"

using namespace std;
using namespace Mini;

void test_shape_h(){
    vector<int> rc = {3, 8, 8};
    Shape one(rc);
    cout << one.dimn() << endl;

    Shape two({2, 6, 8});
    cout << "test TWO" << endl;
    two.change_dim({2, 4, 12});
    cout << two.get_dim(1) << endl;
    two.change_dim({2, 8, -1});
    cout << two.get_dim(2) << "sould be " << "6"  <<endl;
    
    Shape three(two);
    cout << two.get_dim(2) << ' ' << three.get_dim(0) <<endl;

}


void test_tensor_h(){
    Tensor<int> one({2,4,4});
    cout << one.dimn() << ' ' << one.dim(0) << endl;

    vector<float> arr = {2.0, 3.1, 1.9, 3.2, 5.6, 7.8, 9.2, 5.4, 2.0, 1.5, 6.5, 7.9, 3.5, 3.8, 1.2, 2.3, 1.2, 7.3, 4.6, 5.7, 3.2, 9.0, 8,4, 5.7};
    Tensor<float> two(arr, {3, 2, 4}); 
    cout << "Construct two." << endl;
    cout << "loc[1,1,1]: " << two.at({1,1,1}) << endl;
    two.view({3, 4, -1});
    cout << two.dim(2) << " = 2." << endl;
    
}


void test_matrix_h(){
    Matrix<float> one(4,4);
    //cout << one << endl;
    cout << (one + 3.) << endl;
    Matrix<float> two(4);
    one -= 0.2;
    two *= 1.2;
    cout << (one.rows() == two.cols()) << ": " << "true" << endl;
    cout << one.mean() << ' ' << two.sum() << endl;
    Matrix<float> three = one.adjoint() - 0.3;
    cout << three << endl;
    Matrix<float> four = three / 0.9;
    cout << "Matrix Four:" << ' ' << (four + one) << endl;
}


void test_linear_h(){
    vector<float> tmp = {1.0, 2.1, 2.5, 1.1, 1.6, 0.8};
    shared_ptr<Tensor<float>> one(new Tensor<float>(tmp, {1, 6}));
    tmp[0] = 1.4; tmp[2] = 1.0; tmp[4] = 0.9;
    shared_ptr<Tensor<float>> two(new Tensor<float>(tmp, {1, 6}));
    Batch<float> batch1;
    batch1.push(one); batch1.push(two);
    nn::Linear layer(6, 4);
    layer.forward(batch1);
    auto res = layer.results();
    for(int i=0;i<res.batch_num();i++){
        res[i].print();
    }
}

int main(){

    test_linear_h();

    return 0;
}
