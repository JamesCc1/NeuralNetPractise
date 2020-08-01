#ifndef _CROSS_ENTROPY_
#define _CROSS_ENTROPY_

#include "Operator.h"
#include <cmath>

#define FLT_MIN 1.175494e-10F
#define FLT_MAX 3.402823e+18F

namespace Mini{

namespace nn{

class CrossEntropyLoss: public Operator<F>{

public:
    static std::unique_ptr<Operator<F>> Construct(){
        std::unique_ptr<Operator<F>> ptr(new CrossEntropyLoss());
        return ptr;
    }

    virtual void forward(Batch<F>& ford){
        assert(ford.batch_num()>0);
        assert(ford[0].dim(0) == 1 && ford[0].dim(1) > 1);
        assert(labels != nullptr);
        forward_res.clear();

        F loss = 0.0;
        for(int i=0; i<ford.batch_num(); i++){
            auto input = ford[i];
            //std::cout << input << std::endl;
            F val = input.at((*labels)[i].at(0));
            //std::cout << val << ':' << constrain(val) << std::endl;
            loss = -log(constrain(val));
            //std::cout << (*labels)[i].at(0) << std::endl;
            std::shared_ptr<Tensor<F>> res(new Tensor<F>({1}));
            (*res)(0) = loss;
            forward_res.push(res);
        }

    }

    virtual void backward(Batch<F>& back){
        assert(back.batch_num() >0);
        backward_res.clear();
        auto io_dim = (*input_data)[0].dim(1);

        for(int i=0; i< input_data->batch_num(); i++){
            std::shared_ptr<Tensor<F>> res(new Tensor<F>({1, io_dim}));
            int activate = (*labels)[i].at(0);
            F val = (*input_data)[i].at(activate);
            (*res)(activate) = - back[i].at(0) / constrain(val);
            //std::cout << *res << std::endl;

            backward_res.push(res);
        }
    }

private:
    CrossEntropyLoss(){}

    // Do this to avoid 0-value
    F constrain(F val){
        if(val < FLT_MIN) val = FLT_MIN;
        else if(val > FLT_MAX) val = 1e20;
        return val;
    }
};
}

}

#endif
