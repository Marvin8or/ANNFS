#include "Operations.h"

void LinearAlgebra::Operations::CalculateInputToNextLayer(Matrix* w_L, Matrix* a_L_1, Matrix* b_L, Matrix* z_L)
{
	Matrix* tmp = new Matrix(w_L->getNumRows(), a_L_1->getNumCols(), false);
	MultiplyMatrices(w_L, a_L_1, tmp);
	AddMatrices(tmp, b_L, z_L);
	delete tmp;
}

void LinearAlgebra::Operations::MultiplyMatrices(Matrix* a, Matrix* b, Matrix* c)
{
	if (a->getNumCols() != b->getNumRows())
	{
		cerr << "A_rows: " << a->getNumCols() << " != " << "B_rows: " << b->getNumRows() << endl;
		assert(false);
	}

	for (int r_a = 0; r_a < a->getNumRows(); r_a++)
	{
		for (int c_b = 0; c_b < b->getNumCols(); c_b++)
		{
			for (int r_b = 0; r_b < b->getNumRows(); r_b++)
			{
				double multVal = a->getValue(r_a, r_b) * b->getValue(r_b, c_b);
				double newVal = multVal + c->getValue(r_a, c_b);
				c->setValue(r_a, c_b, newVal);
			}
		}
	}
}

void LinearAlgebra::Operations::AddMatrices(Matrix* a, Matrix* b, Matrix* c)
{
	if (a->getNumRows() != b->getNumRows() ||
		a->getNumRows() != c->getNumRows() ||
		b->getNumRows() != c->getNumRows())
	{
		cerr << "Rows of provided matrices are invalid!" << endl;
		assert(false);
	}
	if (a->getNumCols() != b->getNumCols() ||
		a->getNumCols() != c->getNumCols() ||
		b->getNumCols() != c->getNumCols())
	{
		cerr << "Columns of provided matrices are invalid!" << endl;
		assert(false);
	}

	for (int r_a = 0; r_a < a->getNumRows(); r_a++)
	{
		for (int c_a = 0; c_a < a->getNumCols(); c_a++)
		{
			double aValue = a->getValue(r_a, c_a);
			double bValue = b->getValue(r_a, c_a);
			c->setValue(r_a, c_a, aValue + bValue);
		}
	}

}

void LinearAlgebra::Operations::SubtractMatrices(Matrix* a, Matrix* b, Matrix* c)
{
	if (a->getNumRows() != b->getNumRows() ||
		a->getNumRows() != c->getNumRows() ||
		b->getNumRows() != c->getNumRows())
	{
		cerr << "Rows of provided matrices are invalid!" << endl;
		assert(false);
	}
	if (a->getNumCols() != b->getNumCols() ||
		a->getNumCols() != c->getNumCols() ||
		b->getNumCols() != c->getNumCols())
	{
		cerr << "Columns of provided matrices are invalid!" << endl;
		assert(false);
	}

	for (int r_a = 0; r_a < a->getNumRows(); r_a++)
	{
		for (int c_a = 0; c_a < a->getNumCols(); c_a++)
		{
			double aValue = a->getValue(r_a, c_a);
			double bValue = b->getValue(r_a, c_a);
			c->setValue(r_a, c_a, aValue - bValue);
		}
	}
}

void LinearAlgebra::Operations::HadamardProduct(Matrix* a, Matrix* b, Matrix* c)
{
	if (a->getNumRows() != b->getNumRows() ||
		a->getNumRows() != c->getNumRows() ||
		b->getNumRows() != c->getNumRows())
	{
		cerr << "Rows of provided matrices are invalid!" << endl;
		assert(false);
	}
	if (a->getNumCols() != b->getNumCols() ||
		a->getNumCols() != c->getNumCols() ||
		b->getNumCols() != c->getNumCols())
	{
		cerr << "Columns of provided matrices are invalid!" << endl;
		assert(false);
	}

	for (int r_a = 0; r_a < a->getNumRows(); r_a++)
	{
		for (int c_a = 0; c_a < a->getNumCols(); c_a++)
		{
			double aValue = a->getValue(r_a, c_a);
			double bValue = b->getValue(r_a, c_a);
			c->setValue(r_a, c_a, aValue * bValue);
		}
	}

}
