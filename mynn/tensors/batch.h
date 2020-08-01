
#ifndef _BATCH_H_
#define _BATCH_H_


#include "shape.h"
#include "tensor.h"
#include <vector>
#include <memory>


namespace Mini{

template<typename Type>
class Batch{
public:
    using Elemtype = Tensor<Type>;
    using pointer = std::shared_ptr<Elemtype>;
    using Batchtype = std::vector<pointer>;
    using Cordtype = std::initializer_list<int>;


    Batch() = default;

    Batch(const Batch<Type>& other): batch_data(other.batch_data){
    }

    /*inline void push(Elemtype& _tensor){
        if(batch_data.size() > 0){
            if(! _tensor.isSameShape(*(batch_data[0])) )
                throw "Input tensor not match in shape.";
        }
        
    }*/


    inline void push(const pointer p_t){
        if(batch_data.size() > 0){
            if(! p_t->isSameShape(*(batch_data[0])) )
                throw "Input tensor not match in shape.";
        }
        batch_data.push_back(p_t);
    }

    inline void push(const Elemtype& tes){
        if(batch_data.size() > 0){
            if(! tes.isSameShape(*(batch_data[0])) )
                throw "Input tensor not match in shape.";
        }
        pointer p_t(new Elemtype(tes));
        batch_data.push_back(p_t);
    }

    inline void push(Elemtype&& tes){
        if(batch_data.size() > 0){
            if(! tes.isSameShape(*(batch_data[0])) )
                throw "Input tensor not match in shape.";
        }
        pointer p_t(new Elemtype(std::move(tes)));
        batch_data.push_back(p_t);
    }


    inline int batch_num() { return batch_data.size(); }

    Tensor<Type>& operator[](unsigned int index){
        if(index >= batch_data.size()) throw "Index out of range For batch.";
        return *(batch_data[index]);
    }

    void clear(){
        for(int i=0; i<batch_data.size(); i++){
            batch_data[i] = nullptr;
        }
        batch_data.resize(0);
    }

private:
    Batchtype batch_data;
    

};

}
#endif
