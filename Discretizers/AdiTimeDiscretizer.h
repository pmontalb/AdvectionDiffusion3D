
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
	template<typename Real>
	class AdiTimeDiscretizer final: public ITimeDiscretizer<Real>
	{
	public:
		AdiTimeDiscretizer(const InputData<Real>& inputData, const AdiSpaceDiscretizer<Real>& spaceDiscretizer);

		bool Precompute() noexcept override;
		void Compute(Eigen::VectorX<Real>& out, const Eigen::VectorX<Real>& in) noexcept override;

	private:
		const InputData<Real>& _inputData;
		const AdiSpaceDiscretizer<Real>& _spaceDiscretizer;

		using TridiagonalMatrices = std::vector<la::TridiagonalMatrix<Real>>;
		std::array<TridiagonalMatrices, 3> _leftOperators {};
		std::array<TridiagonalMatrices, 3> _rightOperators {};

		std::array<Eigen::VectorX<Real>, 3> _solverCache {};
		std::array<Eigen::VectorX<Real>, 3> _dotProductCache {};
		std::array<Eigen::VectorX<Real>, 3> _inputCache {};
		std::array<Eigen::VectorX<Real>, 3> _intermediateCache {};
	};
}	 // namespace pde

extern template class pde::AdiTimeDiscretizer<float>;
extern template class pde::AdiTimeDiscretizer<double>;
