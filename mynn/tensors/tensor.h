#ifndef _TENSOR_H_
#define _TENSOR_H_


#include "shape.h"
#include <random>
#include <cassert>


namespace Mini{

template<typename Type>
class Tensor{
public:
    using Elemtype = std::vector<Type>;
    using Cordtype = std::initializer_list<int>;


    Tensor():shape({1}){
        data.push_back(Type());
    }

    Tensor(const Cordtype& lst):
       shape(lst), data(shape.capacity(), Type()){
    }

    Tensor(const Elemtype& other, const Cordtype& lst):
       shape(lst), data(other){
    }
 
    Tensor(const Tensor<Type>& other):
       shape(other.shape), data(other.data){
     }

    Tensor(Tensor<Type>&& other): 
        shape(std::move(other.shape)), data(std::move(other.data)){
    }

    template<typename Ty>
    friend std::ostream& operator<<(std::ostream& os, const Tensor<Ty>& other);

    inline Tensor<Type>& operator=(const Tensor<Type>& other){
        if(this == &other) return *this;
        shape = other.shape;
        data = other.data;
        return *this;
    }

    inline Tensor<Type>& operator=(Tensor<Type>&& other){
        shape = std::move(other.shape);
        data = std::move(other.data);
        return *this;
    }

    inline Tensor<Type>& operator+=(const Tensor<Type>& other){
        assert(other.shape == shape);
        for(int i=0; i<num_elems(); i++)
            data[i] += other.data[i];
        return *this;
    }

    inline Tensor<Type>& operator-=(const Tensor<Type>& other){
        assert(other.shape == shape);
        for(int i=0; i<num_elems(); i++)
            data[i] -= other.data[i];
        return *this;
    }

    inline Tensor<Type>& operator*=(const Tensor<Type>& other){
        assert(other.shape == shape);
        for(int i=0; i<num_elems(); i++)
            data[i] *= other.data[i];
        return *this;
    }


    /*
      Following is dim-related.
      Mainly in shape.h.
    */
    inline void view(const Cordtype& lst){
        shape.change_dim(lst);
    }

    inline int dim(unsigned int index) const { 
         return shape.get_dim(index); 
    }

    inline std::vector<int> dims() const { return shape.dims(); }

    inline int dimn() const { return shape.dimn(); }

    inline int num_elems() const{ return shape.capacity();}

    inline bool isSameShape(const Tensor& other) const{
        if(shape.dimn() != other.dimn()) return false;
        for(unsigned int i=0; i < shape.dimn(); i++)
            if(shape.get_dim(i) != other.dim(i)) return false;
        return true;
    }


    /**/
    inline Type at(const int& abs_loc) const { return data[abs_loc]; }

    inline Type at(const Cordtype& lst) const{
        auto abs_loc = location(lst);
        return data[abs_loc];
    }
  
    inline Type& operator()(const int& abs_loc){ return data[abs_loc]; }
    inline Type& operator()(const Cordtype& lst){
        auto abs_loc = location(lst);
        return data[abs_loc];
    }

    inline int max_index(){
        int flag = 0;
        Type val = data[0];
        //std::cout << data[0] << ' ';
        for(int i=1; i<shape.capacity(); i++){
            //std::cout << data[i] << ' ';
            if(data[i]>val){
                flag = i; val = data[i];
            }
        }
        //std::cout << std::endl;
        return flag;
    }

    inline void normal_init(Type mean, Type std){
        std::normal_distribution<Type> norm(mean, std);
        std::default_random_engine rad; 
        for(int index=0; index<data.size(); index++)
            data[index] = norm(rad);
    }

    inline void const_init(Type K){
        for(int index=0; index<data.size(); index++)
            data[index] = K;
    }

    inline void print(){
        for(int i=0; i<data.size(); i++){
             std::cout << data[i] <<' ';
             if(i >= 7) std::cout<< " ...";
        }
        std::cout << std::endl;
    }

private:
    Shape shape;
    Elemtype data;

    int location(const Cordtype& lst) const{
        std::vector<int> temp(lst);
        if(temp.size() != shape.dimn()) throw "Cordinate dose not match.";
        int loc = temp[0];
        assert(loc < shape.get_dim(0) && loc > 0);
        for(unsigned int i = 1; i < temp.size() - 1; i++){
            if(temp[i] < 0 || temp[i] >= shape.get_dim(i))
                throw "The input dim not valid.";
            loc = loc * shape.get_dim(i) + temp[i];
        }
        return loc;
    }

};

template<typename Ty>
std::ostream& operator<<(std::ostream& os, const Tensor<Ty>& other){
    for(int i=0; i<other.data.size(); i++){
        os << other.data[i] << ' ';
    }
    return os;
}

}
#endif
