#ifndef _OPERATOR_H_
#define _OPERATOR_H_


#include <iostream>
#include <vector>
#include "batch.h"

namespace Mini{

using F = float;

namespace nn{

template<typename Type>
class Operator{
public:
    // 
    Operator():
       input_data(nullptr),forward_res(), backward_res() { 
        is_train = true; 
    }

    virtual ~Operator(){}

    virtual inline void train(){ is_train = true; }
    virtual inline void eval(){ is_train = false; }

    virtual std::vector< std::pair<Tensor<Type>*, Tensor<Type>*> > get_param(){
        std::vector<std::pair<Tensor<Type>*, Tensor<Type>*>> wg_vec;
        return wg_vec;
    }

    virtual inline Batch<Type>& down2top(){
        return forward_res;
    }

    virtual inline Batch<Type>& top2down(){
        return backward_res;
    }

    virtual void forward(Batch<Type>& ford) {}
    virtual void backward(Batch<Type>& back){}
    virtual void zero_grad(){}


    virtual void set_input_ptr(Batch<Type>* p){ input_data = p; }
    virtual void set_label_ptr(Batch<Type>* p){ labels = p; }


    //virtual Batch<Type>

    //virtual std::vector<Type>& get_weights() = 0;
    //virtual std::vector<Type>& get_grads() = 0;

protected:
    bool is_train;
    
    Batch<Type>* input_data;
    Batch<Type>* labels;
    Batch<Type> forward_res;
    Batch<Type> backward_res;

};

}

}
#endif
