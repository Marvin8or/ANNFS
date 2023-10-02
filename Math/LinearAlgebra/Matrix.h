#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <tuple>
#include <assert.h>
#include <sstream>
#include "utils.h"

using namespace std;


template<class T>
class Matrix
{
	public:

		// Default Constructor
		Matrix<T>() : rows_(1), cols_(1)
		{
			this->data_ = new T[this->rows_ * this->cols_];
		}

		Matrix<T>(const Matrix<T>& matrix); // TODO implement after overloading == operator

		/** Matrix constructor with defined size
		 * \param rows number of rows
		 * \param columns number of columns
		 */
		Matrix<T>(unsigned int rows, unsigned int columns) : rows_(rows), cols_(columns)
		{
			this->data_ = new T[this->rows_ * this->cols_];
		}

		/** Matrix constructor with defined size and random values
		 * \param rows number of rows
		 * \param columns number of columns
		 * \param random fill wit random numbers if True
		 */
		Matrix<T>(unsigned int rows, unsigned int columns, bool random) : Matrix<T>(rows, columns)
		{
			this->fill(random);
		};

		/** Matrix constructor with defined size and initial value
		 * \param rows number of rows
		 * \param columns number of columns
		 * \param data initial value for all elements
		 */
		Matrix<T>(unsigned int rows, unsigned int columns, T data) : Matrix<T>(rows, columns)
		{
			this->fill(data);
		}

		/*
		 * Matrix destructor
		 */
		~Matrix<T>()
		{
			delete[] data_;
		}

		unsigned int getCols() const;
		unsigned int getRows() const;
		void put(unsigned int row, unsigned int column, const T& value);
		T	 get(unsigned int row, unsigned int column) const;

		void fill(const bool& random);
		void fill(const T& value);
		void print(std::ostream& flux) const;

		Matrix<T> add(const Matrix<T>& matrix) const;

		//Matrix* transpose();
		//Matrix* copy();

		//void populate(double value); //Pass lambda
		//void setValue(int row, int column, double value) { values.at(row).at(column) = value; }
		//double getValue(int row, int column) { return values.at(row).at(column); }
		//int getNumRows() { return this->numRows; }
		//int getNumCols() { return this->numCols; }
		//tuple<int, int> getShape();
		//void printToConsole();

	private:
		unsigned int rows_;
		unsigned int cols_;
		T* data_;


};

template <class T> std::ostream& operator<<(std::ostream& flux, const Matrix<T>& m);

#endif

template <class T>
unsigned int Matrix<T>::getCols() const
{
	return cols_;
}

template <class T>
unsigned int Matrix<T>::getRows() const
{
	return rows_;
}

template <class T>
void Matrix<T>::put(unsigned int row, unsigned int column, const T& value)
{
	if (!(row < rows_))
		throw std::invalid_argument("row index must be smaller than number of rows");

	if (!(column < cols_))
		throw std::invalid_argument("çolumn index must be smaller than number of columns");

	data_[row * rows_ + column] = value;
}


template <class T>
T Matrix<T>::get(unsigned int row, unsigned int column) const
{
	if (!(row < rows_))
		throw std::invalid_argument("row index must be smaller than number of rows");

	if (!(column < cols_))
		throw std::invalid_argument("çolumn index must be smaller than number of columns");

	return data_[row * rows_ + column];
}

template <class T>
void Matrix<T>::fill(const T& value)
{
	for (int i = 0; i < this->rows_; i++)
	{
		for (int j = 0; j < this->cols_; j++)
		{
			this->put(i, j, value);
		}
	}
}

template <class T>
void Matrix<T>::fill(const bool& random)
{
	for (unsigned int i = 0; i < this->rows_; i++)
	{
		for (unsigned int j = 0; j < this->cols_; j++)
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

	for (unsigned int i = 0; i < rows_; i++)
	{
		for (unsigned int j = 0; j < cols_; j++)
		{
			ss << this->get(i, j);

			if (maxLength[j] < ss.str().size())
			{
				maxLength[j] = ss.str().size();
			}

			ss.str(std::string());
		}
	}

	for (unsigned int i = 0; i < rows_; i++)
	{
		for (unsigned int j = 0; j < cols_; j++)
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

	Matrix<T> result(rows_, cols_);
	for(auto i=0; i<rows_; i++)
	{
		for(auto j=0; j<cols_; j++)
		{
			result.put(i, j, this->get(i, j) + matrix.get(i, j));
		}
	}

	return result;
}
template <class T>
std::ostream& operator<<(std::ostream& flux, const Matrix<T>& m) {
	m.print(flux);
	return flux;
}




