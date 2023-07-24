#pragma once
#include "Matrix.h"
namespace LinearAlgebra
{
	class Operations
	{
	public:
		static void CalculateInputToNextLayer(Matrix* w_L, Matrix* a_L_1, Matrix* b_L, Matrix* z_L);
		static void MultiplyMatrices(Matrix* a, Matrix* b, Matrix* c);
		static void AddMatrices(Matrix* a, Matrix* b, Matrix* c);
		static void SubtractMatrices(Matrix* a, Matrix* b, Matrix* c);
		static void HadamardProduct(Matrix* a, Matrix* b, Matrix* c);
	};
}
