
#pragma once

#include "ISpaceDiscretizer.h"
#include "Pde/InputData.h"
#include "Pde/Index.h"
#include "Pde/SolverType.h"

#include <Eigen/Eigen>

namespace pde
{
	template<typename Real>
	class CenteredDifferenceSpaceDiscretizer final: public ISpaceDiscretizer<Real>
	{
	public:
		CenteredDifferenceSpaceDiscretizer(const InputData<Real>& inputData, const std::array<std::size_t, 3>& nSpacePoints, const SolverType solverType)
			: _inputData(inputData), _nSpacePoints(nSpacePoints), _solverType(solverType)
		{
		}

		[[nodiscard]] const auto& GetOperator() const noexcept { return _operator; }

		bool Compute() noexcept override;

	private:
		const InputData<Real>& _inputData;
		const std::array<std::size_t, 3>& _nSpacePoints;
		const SolverType _solverType;

		using SparseMatrix = Eigen::SparseMatrix<Real>;
		SparseMatrix _operator {};
	};
}	 // namespace pde

extern template class pde::CenteredDifferenceSpaceDiscretizer<float>;
extern template class pde::CenteredDifferenceSpaceDiscretizer<double>;
