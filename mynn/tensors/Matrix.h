#ifndef _MATRIX_H_
#define _MATRIX_H_

/*
   a wrapper for Eigen::Matrix
   For convinience.
*/

#include <iostream>
#include <eigen3/Eigen/Dense>


namespace Mini{

template<typename Type>
class Matrix{
public:
    template<typename T>
    using RowMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    
    using Index = Eigen::Index;
    
    /* Costructor
    */
    Matrix(): mat() {}

    Matrix(Index m): mat(m,m) {}
    Matrix(Index m, Index n): mat(m,n) {}

    Matrix(RowMatrix<Type>&& _mat): mat(std::move(_mat)) {}
    Matrix(const RowMatrix<Type>& _mat): mat(_mat) {}

    Matrix(Matrix<Type>&& other): mat(std::move(other.mat)) {}

    Matrix(const Matrix<Type>&) = default;
    
    /* Operator
    */

    Matrix<Type>& operator=(const Matrix<Type>& other){
        if(this == &other) return *this;
        mat = other.mat;
        return *this;
    }

    Matrix<Type>& operator=(Matrix<Type>&& other){
        if(this == &other) throw "Try to assign r-value.";
        mat = std::move(other.mat);
        return *this;
    }


    Matrix<Type> operator+(Type scalar) const{
        RowMatrix<Type> tmp = mat;
        for(int i=0; i<tmp.rows(); i++){
            for(int j=0; j<tmp.cols(); j++)
                tmp(i, j) += scalar;
        }
        return Matrix<Type>(std::move(tmp));
    }

    Matrix<Type>& operator+=(Type scalar){
        for(int i=0; i<mat.rows(); i++){
            for(int j=0; j<mat.cols(); j++)
                mat(i, j) += scalar;
        }
        return *this;
    }


    Matrix<Type> operator-(Type scalar) const{
        RowMatrix<Type> tmp = mat;
        for(int i=0; i<tmp.rows(); i++){
            for(int j=0; j<tmp.cols(); j++)
                tmp(i, j) -= scalar;
        }
        return Matrix<Type>(std::move(tmp));
    }

    Matrix<Type>& operator-=(Type scalar){
        for(int i=0; i<mat.rows(); i++){
            for(int j=0; j<mat.cols(); j++)
                mat(i, j) -= scalar;
        }
        return *this;
    }


    Matrix<Type> operator*(Type scalar) const{
        RowMatrix<Type> tmp = mat;
        for(int i=0; i<tmp.rows(); i++){
            for(int j=0; j<tmp.cols(); j++)
                tmp(i, j) *= scalar;
        }
        return Matrix<Type>(std::move(tmp));
    }

    Matrix<Type>& operator*=(Type scalar){
        for(int i=0; i<mat.rows(); i++){
            for(int j=0; j<mat.cols(); j++)
                mat(i, j) *= scalar;
        }
        return *this;
    }

    Matrix<Type> operator/(Type scalar) const{
        RowMatrix<Type> tmp = mat;
        for(int i=0; i<tmp.rows(); i++){
            for(int j=0; j<tmp.cols(); j++)
                tmp(i, j) /= scalar;
        }
        return Matrix<Type>(std::move(tmp));
    }

    Matrix<Type>& operator/=(Type scalar){
        for(int i=0; i<mat.rows(); i++){
            for(int j=0; j<mat.cols(); j++)
                mat(i, j) /= scalar;
        }
        return *this;
    }


    // op between Eigen::Matrixs

    // TODO: maybe you need broadcast
    Matrix<Type> operator+(const Matrix<Type>& other) const{
        RowMatrix<Type> tmp = mat + other.mat;
        return Matrix<Type>(std::move(tmp));
    }

    Matrix<Type>& operator+=(const Matrix<Type>& other){
        mat += other.mat;
        return *this;
    }

    Matrix<Type> operator-(const Matrix<Type>& other) const{
        RowMatrix<Type> tmp = mat - other.mat;
        return Matrix<Type>(std::move(tmp));
    }

    Matrix<Type>& operator-=(const Matrix<Type>& other){
        mat -= other.mat;
        return *this;
    }

    Matrix<Type> operator*(const Matrix<Type>& other) const{
        RowMatrix<Type> tmp = mat * other.mat;
        return Matrix<Type>(std::move(tmp));
    }

    Matrix<Type>& operator*=(const Matrix<Type>& other){
        mat += other.mat;
        return *this;
    }
    //

    /* Index
    */
    inline Index rows() const { return mat.rows(); }
    inline Index cols() const { return mat.cols(); }
    inline Type at(const Index i, const Index j) const { return mat(i,j); }
    inline Type& operator()(const Index i, const Index j) { return mat(i, j); }

   
    /*
      Base operator.
      Some necessary function about Matrix.
    */

    inline Type sum() const { return mat.sum(); }
    inline Type min() const { return mat.minCoeff(); }
    inline Type max() const { return mat.maxCoeff(); }
    inline Type prod() const { return mat.prod(); }   // All Elements production.
    inline Type mean() const { return mat.mean(); }

    inline void exp(){
        auto tmp = mat.array().exp();
        mat = std::move(tmp.matrix());
    }

    inline Matrix<Type> element_wise(const Matrix<Type>& other){
        Matrix<Type> m(mat);
        for(int i=0; i<m.rows(); i++){
            for(int j=0; j<m.cols(); j++)
                m(i, j) *= other.at(i, j);
        }
        return m;
    }

    inline void swap(Matrix<Type>& other){
        mat.swap(other.mat);
    }

    /* Block */

    Matrix<Type> block(Index i, Index j, Index rows, Index cols) const{
        RowMatrix<Type> tmp = mat.block(i, j, rows, cols);
        return Matrix<Type>(std::move(tmp));    
    }

    Matrix<Type> row(Index i) const{
        RowMatrix<Type> tmp = mat.row(i);
        return Matrix<Type>(std::move(tmp));    
    }

    Matrix<Type> col(Index i) const{
        RowMatrix<Type> tmp = mat.col(i);
        return Matrix<Type>(std::move(tmp));
    }


    /* Matrix necessary 
       transpose,  adjoint
    */
    Matrix<Type> transpose() const{
        RowMatrix<Type> tmp = mat.transpose();
        return Matrix<Type>(std::move(tmp));
    }

    Matrix<Type> adjoint() const{
        RowMatrix<Type> tmp = mat.adjoint();
        return Matrix<Type>(std::move(tmp));
    }

    //  print
    template<typename T>
    friend std::ostream& operator<<(std::ostream& os, const Matrix<T>& other);


private:
    RowMatrix<Type> mat;

};


template <typename T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& other){
    os << other.mat;
    return os;
}

}
#endif
