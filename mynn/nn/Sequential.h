#ifndef _SEQUENTIAL_H_
#define _SEQUENTIAL_H_

#include <vector>
#include <iostream>
#include <memory>
#include "Operator.h"
#include "Optim.h"


namespace Mini{

namespace nn{


class Sequential{
public:
    Sequential(){
        optim = nullptr;
        loss_node = nullptr;
    }

    virtual void add_block(std::unique_ptr<Operator<F>> neuron_node){
        assert(neuron_node != nullptr);
        s_nodes.emplace_back(std::move(neuron_node));
    }

    virtual void set_loss(std::unique_ptr<Operator<F>> loss_func){
        assert(loss_func != nullptr);
        loss_node = std::move(loss_func);
    }

    virtual void forward(Batch<F>& data){
        dynamic_batch = data.batch_num();
        auto input_p = &data;
        for(auto it = s_nodes.begin(); it != s_nodes.end(); ++it){
            //std::cout << "ptr:" << input_p << std::endl;
            //std::cout << "C0 :" << (*input_p)[0] << std::endl;
            (*it)->set_input_ptr(input_p);
            (*it)->forward(*input_p);
            input_p = &((*it)->down2top());
        }
    }

    virtual void backward(){
        assert(loss_node != nullptr);

        auto input = loss_node->top2down();        
        for(auto it = s_nodes.rbegin();it != s_nodes.rend(); ++it){
            (*it)->backward(input);
            input = (*it)->top2down();
        }
    }

    Batch<F>& operator()(Batch<F>& data){
        forward(data);
        return (s_nodes[s_nodes.size()-1])->down2top();
    }

    template<typename T1>
    Batch<F> compute_loss(Batch<F>& data, Batch<T1>& labels){
        assert(s_nodes.size() > 0);
        assert(is_train && loss_node != nullptr);
        loss_node->set_label_ptr(&labels);
        // loss compute
        auto it = s_nodes.end() - 1;
        auto input_p = &((*it)->down2top());
        loss_node->set_input_ptr(input_p);
        loss_node->forward(*input_p);

        auto loss_res = loss_node->down2top();
        loss_node->backward(loss_res);
        return loss_res;
    }


    virtual void use(std::unique_ptr<Optimizer<F>> opt){
        assert(opt!=nullptr);
        optim = std::move(opt);
        assert(s_nodes.size()>0);
        for(int i=0; i<s_nodes.size(); ++i){
            auto param = s_nodes[i]->get_param();
            if(param.empty()) continue;
            for(auto& entity:param){
                //std::cout << entity.first << ' ' << entity.second << std::endl;
                optim->regist(entity);}
        }
    }


    void back_step(){
        backward();
        //std::cout << "back_step one" << std::endl;
        optim->update(dynamic_batch);
    }

    void zero_grad(){
        for(int i=0; i<s_nodes.size(); ++i)
            s_nodes[i]->zero_grad();
    }

    virtual void load_state_dict(){};

    virtual void save_state_dict(){};


    ~Sequential(){}

private:
    std::vector<std::unique_ptr<Operator<F>>> s_nodes;
    bool is_train = true;
    std::unique_ptr<Optimizer<F>> optim;
    std::unique_ptr<Operator<F>> loss_node;

    F dynamic_batch = 1.;
};

}
}

#endif
