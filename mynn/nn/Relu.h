#ifndef _RELU_H_
#define _RELU_H_


#include "Operator.h"

namespace Mini{

namespace nn{

class ReLU: public Operator<F>{
public:
    static std::unique_ptr<Operator<F>> Construct(F leaky){
        std::unique_ptr<Operator<F>> ptr(new ReLU(leaky));
        return ptr;
    }

    virtual void forward(Batch<F>& ford){
        forward_res.clear();
        for(int i=0; i<ford.batch_num(); i++){
            std::shared_ptr<Tensor<F>> ptr(new Tensor<F>(ford[i]));
            auto input = *ptr;
            for(int index=0; index<input.num_elems(); index++){
                if(input.at(index) < 0)
                    input(index) *= leaky_val;
            }
            forward_res.push(ptr);
        }
    }

    virtual void backward(Batch<F>& back){
        backward_res.clear();
        for(int i=0; i<back.batch_num(); i++){
            std::shared_ptr<Tensor<F>> ptr(new Tensor<F>(back[i]));
            auto grad = *ptr;
            auto input_res = forward_res[i];
            for(int index=0; index<grad.num_elems(); index++){
                F K = input_res.at(index) > 0 ? 1.0 : leaky_val;
                grad(index) *= K; 
            }
            backward_res.push(ptr);
        }
    }

private:
    ReLU(F leaky):leaky_val(leaky){}

    F leaky_val;
};


}
}
#endif
