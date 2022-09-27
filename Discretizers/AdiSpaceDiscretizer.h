
#pragma once

#include "ISpaceDiscretizer.h"
#include "Pde/InputData.h"
#include "Pde/Index.h"
#include "Pde/SolverType.h"

#include <Eigen/Eigen>

namespace pde
{
	template<typename Real>
	class AdiSpaceDiscretizer final: public ISpaceDiscretizer<Real>
	{
	public:
		AdiSpaceDiscretizer(const InputData<Real>& inputData, const std::array<std::size_t, 3>& nSpacePoints)
			: _inputData(inputData), _nSpacePoints(nSpacePoints)
		{
		}

		[[nodiscard]] const auto& GetOperator() const noexcept { return _operator; }

		bool Compute() noexcept override;

	private:
		const InputData<Real>& _inputData;

		using SparseMatrix = Eigen::SparseMatrix<Real>;
		SparseMatrix _operator {};
	};
}	 // namespace pde

extern template class pde::AdiSpaceDiscretizer<float>;
extern template class pde::AdiSpaceDiscretizer<double>;
