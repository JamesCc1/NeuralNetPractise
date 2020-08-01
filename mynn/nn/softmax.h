#ifndef _SOFTMAX_H_
#define _SOFTMAX_H_

#include "Operator.h"
#include "utils.h"


namespace Mini{

namespace nn{

class Softmax: public Operator<F>{
public:
    static std::unique_ptr<Operator<F>> Construct(){
        std::unique_ptr<Operator<F>> ptr(new Softmax());
        return ptr;
    }

    virtual void forward(Batch<F>& ford){
        assert(ford.batch_num() > 0);
        assert(ford[0].dim(0) == 1 && ford[0].dim(1) > 1);
        forward_res.clear();
        auto io_dim = ford[0].dim(1);

        for(int i=0; i<ford.batch_num(); i++){
            auto input = ford[i];
            auto input_matrix = Matrix<F>(1, io_dim);
            tensor2matrix<F>(input_matrix, input);
            input_matrix -= input_matrix.max();
            input_matrix.exp();
            input_matrix /= input_matrix.sum();
            std::shared_ptr<Tensor<F>> res(new Tensor<F>({1, io_dim}));
            matrix2tensor(input_matrix, *res);
            //constrain(*res, F(0.001), F(0.999));
            forward_res.push(res);
        }
        //std::cout << std::endl;
    }

    virtual void backward(Batch<F>& back){
        assert(back.batch_num() > 0);
        assert(back[0].dim(0) == 1 && back[0].dim(1) > 1);
        backward_res.clear();
        auto io_dim =  back[0].dim(1);

        for(int i=0; i<back.batch_num(); i++){
            auto grad_back = Matrix<F>(1, io_dim);
            tensor2matrix<F>(grad_back, back[i]);
            auto fp = Matrix<F>(1, io_dim);
            tensor2matrix<F>(fp, forward_res[i]);
            auto res = fp.element_wise(grad_back);
            auto sum_val = res.sum();
            auto grad = res - fp * sum_val;
            std::shared_ptr<Tensor<F>> grad_res(new Tensor<F>({1, io_dim}));
            matrix2tensor<F>(grad, *grad_res);
            backward_res.push(grad_res);
        }
        //std::cout << backward_res[0] << std::endl;
    }

private:
    Softmax(){};
};

}

}
#endif
