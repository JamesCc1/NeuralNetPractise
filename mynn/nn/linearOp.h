#ifndef _LINEAR_H_
#define _LINEAR_H_

#include <cassert>
#include <random>
#include "Operator.h"
#include "tensor.h"
#include "utils.h"


namespace Mini{


namespace nn{


class Linear: public Operator<F>{

public:
    static std::unique_ptr<Operator<F>> Construct(int x, int y){
        std::unique_ptr<Operator<F>> ptr(new Linear(x, y));
        return ptr;
    }


    virtual std::vector< std::pair<Tensor<F>*, Tensor<F>*> > get_param(){
        std::vector<std::pair<Tensor<F>*, Tensor<F>*>> wg_vec;
        wg_vec.push_back({&weights, &grad_weight});
        wg_vec.push_back({&bias, &grad_bias});
        return wg_vec;
    }


    void forward(Batch<F>& ford){
        assert(ford.batch_num() > 0);
        assert(ford[0].dim(0) == 1 && ford[0].dim(1) == input_dim);
        forward_res.clear();
        //std::cout << "linear forward." << std::endl;
        for(int i=0; i<ford.batch_num(); i++){
            auto input = ford[i];
            auto input_matrix = Matrix<F>(1, input_dim);
            tensor2matrix<F>(input_matrix, input);
            auto weight_matrix = Matrix<F>(input_dim, output_dim);
            tensor2matrix<F>(weight_matrix, weights);
            auto bias_matrix = Matrix<F>(1, output_dim);
            tensor2matrix<F>(bias_matrix, bias);
            auto res_matrix = input_matrix*weight_matrix + bias_matrix;
            std::shared_ptr<Tensor<F>> res(new Tensor<F>({1, output_dim}));
            matrix2tensor<F>(res_matrix, *res);
            forward_res.push(res);
        }
    }
        
    void backward(Batch<F>& back){
        //std::cout << "linear back." << std::endl;
        assert(back.batch_num() > 0);
        assert(input_data->batch_num() == back.batch_num());
        assert(back[0].dim(0) == 1 && back[0].dim(1) == output_dim);
        backward_res.clear();

        auto weight_matrix = Matrix<F>(input_dim, output_dim);
        //auto bias_matrix = Matrix<F>(1, output_dim);
        tensor2matrix<F>(weight_matrix, weights);
        //tensor2matrix<F>(bias_matrix, bias);
        //std::cout << grad_weight << std::endl;

        for(int i=0; i<back.batch_num(); i++){
            auto grad_back = Matrix<F>(1, output_dim);
            tensor2matrix<F>(grad_back, back[i]);
            std::shared_ptr<Tensor<F>> res(new Tensor<F>({1, input_dim}));
            auto tmp = grad_back * weight_matrix.transpose();
            matrix2tensor<F>(tmp, *res);
            backward_res.push(res);

            //
            auto x = (*input_data)[i];
            auto input_matrix = Matrix<F>(1, input_dim);
            tensor2matrix<F>(input_matrix, x);
            auto grad_w_matrix = input_matrix.transpose() * grad_back;
            auto grad_w_tensor = Tensor<F>({input_dim, output_dim});
            matrix2tensor<F>(grad_w_matrix, grad_w_tensor);
            //std::cout <<  grad_w_matrix << std::endl;
            auto grad_bias_tensor = Tensor<F>({1, output_dim});
            matrix2tensor<F>(grad_back, grad_bias_tensor);
            // apply add 
            grad_weight += grad_w_tensor;
            grad_bias += grad_bias_tensor;

        }
    }

    
    virtual void zero_grad(){
        for(int i=0; i<grad_weight.num_elems(); ++i)
            grad_weight(i) = 0.0;
        for(int i=0; i<grad_bias.num_elems(); ++i)
            grad_bias(i) = 0.0;
    }

    

private:
    int input_dim;
    int output_dim;
    Tensor<F> weights;
    Tensor<F> bias;
    Tensor<F> grad_weight;
    Tensor<F> grad_bias;
    

    Linear(int input, int output): 
        input_dim(input), output_dim(output),
        weights({input, output}),
        bias({1, output}),
        grad_weight({input, output}),
        grad_bias({1, output})
    {
        _init(); 
    }

    inline void _init(){
        weights.normal_init(F(0.0), 1./static_cast<F>(input_dim));
        bias.const_init(F(0.0));
    }


};

}

} 
#endif
