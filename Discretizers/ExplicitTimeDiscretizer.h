
#pragma once

#include "CenteredDifferenceSpaceDiscretizer.h"
#include "ITimeDiscretizer.h"
#include "Pde/Index.h"
#include "Pde/InputData.h"
#include "Pde/SolverType.h"

#include <Eigen/Eigen>

namespace pde
{
	template<typename Real>
	class ExplicitTimeDiscretizer final: public ITimeDiscretizer<Real>
	{
	public:
		ExplicitTimeDiscretizer(const InputData<Real>& inputData, const CenteredDifferenceSpaceDiscretizer<Real>& spaceDiscretizer) : _inputData(inputData), _spaceDiscretizer(spaceDiscretizer) {}

		bool Precompute() noexcept override;
		void Compute(Eigen::VectorX<Real>& out, const Eigen::VectorX<Real>& in) noexcept override;

	private:
		const InputData<Real>& _inputData;
		const CenteredDifferenceSpaceDiscretizer<Real>& _spaceDiscretizer;

		using SparseMatrix = Eigen::SparseMatrix<Real>;
		SparseMatrix _operator {};
	};
}	 // namespace pde

extern template class pde::ExplicitTimeDiscretizer<float>;
extern template class pde::ExplicitTimeDiscretizer<double>;
