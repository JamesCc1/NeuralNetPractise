#ifndef _CC_UTIL_H_
#define _CC_UTIL_H_


#include "tensor.h"
#include "Matrix.h"
#include <cassert>
#include <eigen3/Eigen/Dense>


namespace Mini{

template<typename Type>
void matrix2tensor(const Matrix<Type>& mat, Tensor<Type>& tes){
    auto col_num = mat.cols();
    //std::cout << _num << ' ' <<mat.cols() << ' '<< tes.dim(0)<< ' ' << tes.num_elems() << std::endl;
    //std::cout << tes << std::endl;
    assert(mat.rows() == tes.dim(0) && (col_num * mat.rows()) == tes.num_elems());
    for(int i=0; i<mat.rows(); i++){
        for(int j=0; j<col_num; j++)
            tes(i*col_num+j) = mat.at(i,j);
    }
}


template<typename Type>
void constrain(Tensor<Type>& tes, Type low, Type high){
    assert(low<high);
    for(int i=0; i<tes.num_elems(); ++i){
        if(tes(i) < low) tes(i) = low;
        if(tes(i) > high) tes(i) = high;
    }
}



template<typename Type>
void tensor2matrix(Matrix<Type>& mat, const Tensor<Type>& tes){
    auto col_num = mat.cols();
    assert(mat.rows() == tes.dim(0) && (col_num * mat.rows()) == tes.num_elems() );
    for(int i=0; i<mat.rows(); i++){
        for(int j=0; j<col_num; j++)
            mat(i, j) = tes.at(i*col_num+j);
    }
}


}
#endif
