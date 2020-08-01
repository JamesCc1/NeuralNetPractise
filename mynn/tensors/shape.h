#ifndef _SHAPE_H_
#define _SHAPE_H_

#include <iostream>
#include <vector>



namespace Mini{

class Shape{
public:
    Shape() = delete;

    Shape(const std::initializer_list<int>& lst):max_elems(1){
        for(auto x:lst){
            shapes.push_back(x);
            max_elems *= x;
        }
    }

    Shape(const std::vector<int>& lst):max_elems(1){
        for(auto x:lst){
            shapes.push_back(x);
            max_elems *= x;
        }
    }

    Shape(const Shape& other): 
        shapes(other.shapes), max_elems(other.max_elems) {
    }


    inline int get_dim(unsigned int index) const{
        if(index >= shapes.size()) throw "Index out of range.";
        return shapes[index];
    }

    inline int dimn() const{ return shapes.size(); }

    inline std::vector<int> dims() const{ return shapes; }

    inline int capacity() const { return max_elems; }

    void change_dim(const std::initializer_list<int>& lst){
        int remark = -1, count = 0;
        std::vector<int> temp(lst);
        int _max_elems = max_elems;
        for(auto x:temp){
            if((remark != -1 && x == -1) || x == 0 || x < -1) throw "Input dim list not valid.";
            if(x == -1) { remark = count ++; continue; }
            if(_max_elems % x != 0) throw "Dim cant be divided well.";
            _max_elems /= x;
            count ++;
        }
        if(remark != -1) temp[remark] = _max_elems;

        int pre_size = shapes.size();
        shapes = temp;
    }

    friend bool operator==(Shape one, Shape other);


private:
    std::vector<int> shapes;
    int max_elems;
};


bool operator==(Shape one, Shape other){
    if(one.dimn() != other.dimn()) return false;
    for(int i=0; i<one.dimn(); ++i)
        if(one.get_dim(i) != other.get_dim(i)) return false;
    return true;
}

}
#endif
