#ifndef _SGD_H_
#define _SGD_H_

#include <unordered_map>
#include <typeinfo>
#include "Optim.h"

namespace Mini{


namespace nn{

class SGD: public Optimizer<F>{

public:
    static std::unique_ptr<Optimizer<F>> Construct(F lr, F momentum=0.5, F weight_decay=0.0){
        std::unique_ptr<Optimizer<F>> ptr(new SGD(lr, momentum, weight_decay));
        return ptr;
    }

    virtual inline void regist(Tensor<F>* w_addr, Tensor<F>* grad_addr){
        auto cc = std::make_pair(w_addr, grad_addr);
        content.push_back(cc);
        Tensor<F> pre(*grad_addr);
        pre_grad[w_addr] = pre;
    }

    virtual inline void regist(std::pair<Tensor<F>*, Tensor<F>*> w_and_grad){
        content.push_back(w_and_grad);
        Tensor<F> pre(*(w_and_grad.second));
        pre_grad[w_and_grad.first] = pre;
    }

    virtual void show(){
        for(auto& uu:content){
           std::cout << "show: " << std::endl;
           std::cout << uu.first << ' ' << uu.second << std::endl;
        }
    }

    virtual void update(F batch_num){
        // for each pair<weight and its grad_mat> update the weight
        for(auto& cnt:content){
            Tensor<F>* const weight_p = cnt.first;
            Tensor<F>* const grad_p = cnt.second;
            assert(weight_p->dimn()==grad_p->dimn() && weight_p->num_elems()==grad_p->num_elems());
            Tensor<F>& pre = pre_grad[weight_p];
            for(int i=0; i<weight_p->num_elems(); ++i){
                F d_p = grad_p->at(i) / batch_num;
                // L2 regularization.
                d_p += _weight_decay * weight_p->at(i) / batch_num;
                F V = _momentum * pre.at(i) + d_p;
                (*weight_p)(i) -= learning_rate * V;
                pre(i) = V;
            }
        }
        // for each weight val -- gradient decent
    }

private:
    F _momentum;
    F _weight_decay;

    std::unordered_map<Tensor<F>*, Tensor<F>> pre_grad;


    SGD(F lr, F momentum, F weight_decay): 
       Optimizer(lr), _momentum(momentum), _weight_decay(weight_decay)
    {}

};

}
}

#endif
