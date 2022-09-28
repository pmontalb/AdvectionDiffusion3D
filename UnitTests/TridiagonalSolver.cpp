
#include <gtest/gtest.h>

#include "LinearAlgebra/TridiagonalSolver.h"

class TridiagonalSolverTests: public testing::Test
{
};

TEST_F(TridiagonalSolverTests, Identity)
{
	// single precision
	{
		la::TridiagonalMatrix<float> A(10);
		A.SetIdentity();

		Eigen::VectorX<float> x;
		x.resize(static_cast<int>(A.Size()));
		x.setRandom();

		la::TridiagonalSolver<float> solver(A);

		Eigen::VectorX<float> y;
		y = x;
		solver.Solve(y);

		for (int i = 0; i < static_cast<int>(A.Size()); ++i)
		{
			ASSERT_FLOAT_EQ(x[i], y[i]);
		}
	}

	// double precision
	{
		la::TridiagonalMatrix<double> A(10);
		A.SetIdentity();

		Eigen::VectorX<double> x;
		x.resize(static_cast<int>(A.Size()));
		x.setRandom();

		la::TridiagonalSolver<double> solver(A);

		Eigen::VectorX<double> y;
		y = x;
		solver.Solve(y);

		for (int i = 0; i < static_cast<int>(A.Size()); ++i)
		{
			ASSERT_DOUBLE_EQ(x[i], y[i]);
		}
	}
}

TEST_F(TridiagonalSolverTests, Regression)
{
	// single precision
	{
		la::TridiagonalMatrix<float> A(10);
		A.SetIdentity();
		for (size_t i = 0; i < A.Size(); ++i)
			A[i] = { std::sin(0.5f * static_cast<float>(i) + 1), std::cos(0.5f * static_cast<float>(i) + 1), std::exp(0.05f * static_cast<float>(i) + 1) };

		Eigen::VectorX<float> x;
		x.resize(static_cast<int>(A.Size()));
		x.setRandom();

		la::TridiagonalSolver<float> solver(A);

		Eigen::VectorX<float> y;
		y = x;
		solver.Solve(y);

		const auto z = A.Dot(y);
		for (int i = 0; i < static_cast<int>(A.Size()); ++i)
		{
			EXPECT_NEAR(x[i], z[i], 7e-5) << i;
		}
	}

	// double precision
	{
		la::TridiagonalMatrix<double> A(10);
		A.SetIdentity();
		for (size_t i = 0; i < A.Size(); ++i)
			A[i] = { std::sin(0.5 * static_cast<double>(i) + 1), std::cos(0.5 * static_cast<double>(i) + 1), std::exp(0.05 * static_cast<double>(i) + 1) };

		Eigen::VectorX<double> x;
		x.resize(static_cast<int>(A.Size()));
		x.setRandom();

		la::TridiagonalSolver<double> solver(A);

		Eigen::VectorX<double> y;
		y = x;
		solver.Solve(y);

		const auto z = A.Dot(y);
		for (int i = 0; i < static_cast<int>(A.Size()); ++i)
		{
			EXPECT_NEAR(x[i], z[i], 1.72e-13) << i;
		}
	}
}
