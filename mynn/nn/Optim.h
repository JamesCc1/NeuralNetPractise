#ifndef _OPTIMIZER_H_
#define _OPTIMIZER_H_


#include <vector>
#include <utility>
#include "tensor.h"

namespace Mini{

using F = float;

namespace nn{

template<typename Type>
class Optimizer{
public:
    virtual void update(F batch_num){};

    // load the update weight and grad
    virtual inline void regist(Tensor<F>* w_addr, Tensor<F>* grad_addr) {}
    virtual inline void regist(std::pair<Tensor<F>*, Tensor<F>*> w_and_grad) {}

    virtual void show(){}
    // set and get the learning rate
    virtual Type get_lr() const { return learning_rate; }

    virtual inline void set_lr(Type val){
        learning_rate = val;
    }

    virtual ~Optimizer(){}


protected:
    Optimizer(Type lr):learning_rate(lr){}

    std::vector< std::pair<Tensor<Type>*, Tensor<Type>*> > content; 
    Type learning_rate;

};
}

}
#endif
