#pragma once
#include "Matrix.h"
namespace LinearAlgebra
{
	class Operations
	{
	public:
		static void MultiplyMatrices(Matrix* a, Matrix* b, Matrix* c);
		static void SubtractMatrices(Matrix* a, Matrix* b, Matrix* c);
		static void HadamardProduct(Matrix* a, Matrix* b, Matrix* c);
	};
}
