#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <tuple>
#include <assert.h>
#include <sstream>
#include <initializer_list>
#include "utils.h"

using namespace std;
typedef unsigned int uint;

template<class T>
class Matrix
{
	public:

		Matrix<T>(std::initializer_list<std::initializer_list<T>> initList);

		Matrix<T>(const Matrix<T>& matrix) : rows_(matrix.getRows()), cols_(matrix.getCols())
		{

			data_.reset(new T[rows_ * cols_]);
			for (uint i = 0; i<rows_; i++)
			{
				for (uint j = 0; j < this->cols_; j++)
				{
					this->put(i, j, matrix.get(i, j));
				}
			}
		}

		/** Matrix constructor with defined size and random values
		 * \param rows number of rows
		 * \param columns number of columns
		 * \param random fill wit random numbers if True
		 */
		Matrix<T>(uint rows, uint columns) : rows_(rows), cols_(columns)
		{
			data_.reset(new T[rows_ * cols_]);
			this->fillRandom();
		};

		/** Matrix constructor with defined size and initial value
		 * \param rows number of rows
		 * \param columns number of columns
		 * \param data initial value for all elements
		 */
		Matrix<T>(uint rows, uint columns, T data) : rows_(rows), cols_(columns)
		{
			data_.reset(new T[rows_ * cols_]);
			this->fill(data);
		}

		/*
		 * Matrix destructor
		 */
		~Matrix<T>()
		{
			data_.reset();
		}

		uint getCols() const;
		uint getRows() const;
		void put(uint row, uint column, const T& value);
		T	 get(uint row, uint column) const;

		void fillRandom();
		void fill(const T& value);
		void print(std::ostream& flux) const;

		Matrix<T> add(const Matrix<T>& matrix) const;
		Matrix<T> subtract(const Matrix<T>& matrix) const;
		Matrix<T> multiply(const Matrix<T>& matrix) const;
		Matrix<T> multiply(const T& value) const;
		Matrix<T> divide(const Matrix<T>& matrix) const;
		Matrix<T> divide(const T& value) const;
		Matrix<T> dot(const Matrix<T>& matrix) const;
		Matrix<T> transpose() const;
		Matrix<T> applyFunction(T(*function)(T)) const;

		bool operator==(const Matrix<T>& matrix);
		bool operator!=(const Matrix<T>& matrix);
		Matrix<T> operator+=(const Matrix<T>& matrix);
		Matrix<T> operator-=(const Matrix<T>& matrix);
		Matrix<T> operator*=(const Matrix<T>& matrix);
		Matrix<T> operator*=(const T& value);
		Matrix<T> operator/=(const Matrix<T>& matrix);
		Matrix<T> operator/=(const T& value);
		T& operator()(uint y, uint x);

	private:

		uint rows_;
		uint cols_;
		std::shared_ptr<T> data_;
};

template <class T> std::ostream& operator<<(std::ostream& flux, const Matrix<T>& m);
template <class T> Matrix<T> operator+(const Matrix<T>& a, const Matrix<T>& b);
template <class T> Matrix<T> operator-(const Matrix<T>& a, const Matrix<T>& b);
template <class T> Matrix<T> operator*(const Matrix<T>& a, const Matrix<T>& b);
template <class T> Matrix<T> operator*(const T& value, const Matrix<T>& b);
template <class T> Matrix<T> operator/(const Matrix<T>& a, const Matrix<T>& b);
template <class T> Matrix<T> operator/(const T& value, const Matrix<T>& b);

#endif

template <class T>
Matrix<T>::Matrix(std::initializer_list<std::initializer_list<T>> initList)
{
	std::shared_ptr<vector<int>> vptr = std::make_shared<vector<int>>();
	auto rows = initList.size();

	for(auto row : initList)
	{
		vptr->push_back((int)row.size());
	}

	auto cols = (*vptr).at(0);
	for(auto rowSize : *vptr)
	{
		if (rowSize != cols)
			throw std::invalid_argument("All rows must have same size!");
	}

	rows_ = (uint)rows;
	cols_ = (uint)cols;
	data_.reset(new T[rows_ * cols_]);

	auto row_i = 0;
	for (auto row : initList)
	{
		auto col_i = 0;
		for (auto col : row)
		{
			this->put(row_i, col_i, col);
			col_i++;
		}
		row_i++;
	}
}

template <class T>
uint Matrix<T>::getCols() const
{
	return cols_;
}

template <class T>
uint Matrix<T>::getRows() const
{
	return rows_;
}

template <class T>
void Matrix<T>::put(uint row, uint column, const T& value)
{
	if (!(row < rows_))
		throw std::invalid_argument("row index must be smaller than number of rows");

	if (!(column < cols_))
		throw std::invalid_argument("çolumn index must be smaller than number of columns");

	data_.get()[row * cols_ + column] = value;
}


template <class T>
T Matrix<T>::get(uint row, uint column) const
{
	if (!(row < rows_))
		throw std::invalid_argument("row index must be smaller than number of rows");

	if (!(column < cols_))
		throw std::invalid_argument("çolumn index must be smaller than number of columns");

	return data_.get()[row * cols_ + column];
}

template <class T>
void Matrix<T>::fill(const T& value)
{
	for (uint i = 0; i < this->rows_; i++)
	{
		for (uint j = 0; j < this->cols_; j++)
		{
			this->put(i, j, value);
		}
	}
}

template <class T>
void Matrix<T>::fillRandom()
{
	for (uint i = 0; i < this->rows_; i++)
	{
		for (uint j = 0; j < this->cols_; j++)
		{
			this->put(i, j, generateRandomNumber());
		}
	}
}

template <class T>
void Matrix<T>::print(std::ostream& flux) const
{
	std::vector<int> maxLength(cols_);
	std::stringstream ss;

	for (uint i = 0; i < rows_; i++)
	{
		for (uint j = 0; j < cols_; j++)
		{
			ss << this->get(i, j);

			if (maxLength[j] < ss.str().size())
			{
				maxLength[j] = ss.str().size();
			}

			ss.str(std::string());
		}
	}

	for (uint i = 0; i < rows_; i++)
	{
		for (uint j = 0; j < cols_; j++)
		{
			flux << this->get(i, j);
			ss << this->get(i, j);

			for (int k = 0; k < maxLength[j] - ss.str().size() + 1; k++) {
				flux << " ";
			}
			ss.str(std::string());
		}
		flux << std::endl;
	}
}
template <class T>
Matrix<T> Matrix<T>::add(const Matrix<T>& matrix) const
{
	if (!(cols_ == matrix.getCols() && rows_ == matrix.getRows()))
		throw std::invalid_argument("Matrices must have same dimensions!");

	Matrix result(rows_, cols_, 0.0);
	for(uint i=0; i<rows_; i++)
	{
		for(uint j=0; j<cols_; j++)
		{
			result.put(i, j, this->get(i, j) + matrix.get(i, j));
		}
	}

	return result;
}

template <class T>
Matrix<T> Matrix<T>::subtract(const Matrix<T>& matrix) const
{
	if (!(cols_ == matrix.getCols() && rows_ == matrix.getRows()))
		throw std::invalid_argument("Matrices must have same dimensions!");

	Matrix result(rows_, cols_, 0.0);
	for (uint i = 0; i < rows_; i++)
	{
		for (uint j = 0; j < cols_; j++)
		{
			result.put(i, j, this->get(i, j) - matrix.get(i, j));
		}
	}

	return result;
}

template <class T>
Matrix<T> Matrix<T>::multiply(const Matrix<T>& matrix) const
{
	if(!(matrix.getRows()==rows_ && matrix.getCols()==cols_))
		throw std::invalid_argument("Matrix dimension must be the same.");

	Matrix<T> result(rows_, cols_, 0.0);
	for(uint i = 0; i < rows_; i++)
	{
		for (uint j = 0; j < cols_; j++)
		{
			result.put(i, j, this->get(i, j) * matrix.get(i, j));
		}
	}
	return result;
}

template <class T>
Matrix<T> Matrix<T>::multiply(const T& value) const
{
	Matrix<T> result(rows_, cols_);
	for (uint i = 0; i < rows_; i++)
	{
		for (uint j = 0; j < cols_; j++)
		{
			result.put(i, j, this->get(i, j) * value);
		}
	}
	return result;
}

template <class T>
Matrix<T> Matrix<T>::divide(const Matrix<T>& matrix) const
{
	if (!(matrix.getRows() == rows_ && matrix.getCols() == cols_))
		throw std::invalid_argument("Matrix dimension must be the same.");

	Matrix<T> result(rows_, cols_, 0.0);
	for (uint i = 0; i < rows_; i++)
	{
		for (uint j = 0; j < cols_; j++)
		{
			result.put(i, j, this->get(i, j) / matrix.get(i, j));
		}
	}
	return result;
}

template <class T>
Matrix<T> Matrix<T>::divide(const T& value) const
{
	Matrix<T> result(rows_, cols_);
	for (uint i = 0; i < rows_; i++)
	{
		for (uint j = 0; j < cols_; j++)
		{
			result.put(i, j, this->get(i, j) / value);
		}
	}
	return result;
}

template <class T>
Matrix<T> Matrix<T>::dot(const Matrix<T>& matrix) const
{
	if (!(rows_ == matrix.getCols() && cols_ == matrix.getRows()))
		throw std::invalid_argument("Dot product cannot be calculated! Check dimensions.");

	T value = 0;
	int mCols = matrix.getCols();
	Matrix<T> result(rows_, mCols);

	for(uint a=0; a<rows_; a++)
	{
		for(uint b=0; b<mCols; b++)
		{
			for(uint c=0; c<rows_; c++)
			{
				value += this->get(a, c) * matrix.get(c, b);
			}
			result.put(a, b, value);
			value = 0;
		}
	}

	return result;
}

template <class T>
Matrix<T> Matrix<T>::transpose() const
{
	Matrix<T> result(rows_, cols_, 0.0);
	for(uint i=0; i<rows_; i++)
	{
		for(uint j=0; j<cols_; j++)
		{
			result.put(i, j, this.get(j, i));
		}
	}
	return result;
}

template <class T>
Matrix<T> Matrix<T>::applyFunction(T(*function)(T)) const
{
	Matrix<T> result(rows_, cols_, 0.0);
	for (uint i = 0; i < rows_; i++)
	{
		for (uint j = 0; j < cols_; j++)
		{
			result.put(i, j, (*function)(this.get(i, j)));
		}
	}
	return result;
}

template <class T>
bool Matrix<T>::operator==(const Matrix<T>& matrix)
{
	if(rows_==matrix.getRows() && cols_==matrix.getCols()){
		for(uint i=0; i<rows_; i++){
			for(uint j=0; j<cols_; j++){
				if (!(this->get(i, j) == matrix.get(i, j))){
					return false;
				}
			}
		}
		return true;
	}
	return false;
}

template <class T>
bool Matrix<T>::operator!=(const Matrix<T>& matrix)
{
	return !operator==(matrix);
}

template <class T>
Matrix<T> Matrix<T>::operator+=(const Matrix<T>& matrix)
{
	data_ = add(matrix).data_;
	return *this;
}

template <class T>
Matrix<T> Matrix<T>::operator-=(const Matrix<T>& matrix)
{
	data_ = subtract(matrix).data_;
	return *this;
}

template <class T>
Matrix<T> Matrix<T>::operator*=(const Matrix<T>& matrix)
{
	data_ = multiply(matrix).data_;
	return *this;
}

template <class T>
Matrix<T> Matrix<T>::operator*=(const T& value)
{
	data_ = multiply(value).data_;
	return *this;
}

template <class T>
Matrix<T> Matrix<T>::operator/=(const Matrix<T>& matrix)
{
	data_ = divide(matrix).data_;
	return *this;
}

template <class T>
Matrix<T> Matrix<T>::operator/=(const T& value)
{
	data_ = divide(value).data_;
	return *this;
}

template <class T>
T& Matrix<T>::operator()(uint x, uint y)
{
	if (!(x < rows_ && y < cols_))
		throw std::invalid_argument("arguments out of bounds!");

	return get(x, y);
}

template <class T>
std::ostream& operator<<(std::ostream& flux, const Matrix<T>& m) {
	m.print(flux);
	return flux;
}

template <class T>
Matrix<T> operator+(const Matrix<T>& a, const Matrix<T>& b) {
	return a.add(b);
}

template <class T>
Matrix<T> operator-(const Matrix<T>& a, const Matrix<T>& b) {
	return a.subtract(b);
}

template <class T>
Matrix<T> operator*(const Matrix<T>& a, const Matrix<T>& b) {
	return a.multiply(b);
}

template <class T>
Matrix<T> operator*(const T& value, const Matrix<T>& b){
	return b.multiply(value);
}

template <class T>
Matrix<T> operator/(const Matrix<T>& a, const Matrix<T>& b) {
	return a.divide(b);
}

template <class T>
Matrix<T> operator/(const T& value, const Matrix<T>& b){
	return b.divide(value);
}

