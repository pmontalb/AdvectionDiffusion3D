
#pragma once

#include "AdiSpaceDiscretizer.h"
#include "ITimeDiscretizer.h"
#include "Pde/Index.h"
#include "Pde/InputData.h"
#include "Pde/SolverType.h"

#include "LinearAlgebra/TridiagonalMatrix.h"
#include "LinearAlgebra/TridiagonalSolver.h"

namespace pde
{
	/*
	 * Douglas-Gunn Alternating-Direction Implicit
	 * https://web.njit.edu/~matveev/documents/adi.pdf
	 *
	 * f is a time-invariant source term
	 *
	 * (I - 0.5 * dt * Lx) * C_X = (I + 0.5 * dt * Lx) * C_{n - 1} + dt * (L_y + L_z) * C_{n - 1} + dt * f
	 * (I - 0.5 * dt * Ly) * C_Y = (I + 0.5 * dt * Ly) * C_{n - 1} + dt * (0.5 * L_x + L_z) * C_{n - 1} + dt * L_x * C_X + dt * f
	 * (I - 0.5 * dt * Lz) * C_n = (I + 0.5 * dt * Lz) * C_{n - 1} + 0.5 * dt * (L_x + L_y) * C_{n - 1} + 0.5 * dt * L_x * (C_X + C_Y) + dt * f
	 *
	 * +++ Unconditionally stable
	 * --- Subject to splitting error
	 * */
	template<typename Real>
	class AdiTimeDiscretizer final: public ITimeDiscretizer<Real>
	{
	public:
		AdiTimeDiscretizer(const InputData<Real>& inputData, const AdiSpaceDiscretizer<Real>& spaceDiscretizer);

		bool Precompute() noexcept override;
		void Compute(Eigen::VectorX<Real>& out, const Eigen::VectorX<Real>& in) noexcept override;
		void Compute(Eigen::VectorX<Real>& out, const Eigen::VectorX<Real>& in, const Eigen::VectorX<Real>& sourceTerm) noexcept override;

	private:
		void ComputeWorker(Eigen::VectorX<Real>& out, const Eigen::VectorX<Real>& in, const Eigen::VectorX<Real>* sourceTerm) noexcept;

	private:
		const InputData<Real>& _inputData;
		const AdiSpaceDiscretizer<Real>& _spaceDiscretizer;

		using TridiagonalMatrices = std::vector<la::TridiagonalMatrix<Real>>;
		std::array<TridiagonalMatrices, 3> _leftOperators {};
		std::array<TridiagonalMatrices, 3> _rightOperators {};

		// Tridiagonal system solve cache
		std::array<Eigen::VectorX<Real>, 3> _solverCache {};

		// Tridiagonal dot product cache
		std::array<Eigen::VectorX<Real>, 3> _dotProductCache {};

		// Cache the relevant dimension from the input
		std::array<Eigen::VectorX<Real>, 3> _inputCache {};

		// store the intermediate result
		Eigen::VectorX<Real> _outCacheX;
		Eigen::VectorX<Real> _outCacheY;

		Eigen::VectorX<Real> _previousSourceTerm;
	};
}	 // namespace pde

extern template class pde::AdiTimeDiscretizer<float>;
extern template class pde::AdiTimeDiscretizer<double>;
