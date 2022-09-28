
#pragma once

#include "ISpaceDiscretizer.h"
#include "Pde/Index.h"
#include "Pde/InputData.h"
#include "Pde/SolverType.h"

#include "LinearAlgebra/TridiagonalMatrix.h"

namespace pde
{
	template<typename Real>
	class AdiSpaceDiscretizer final: public ISpaceDiscretizer<Real>
	{
	public:
		AdiSpaceDiscretizer(const InputData<Real>& inputData, const std::array<std::size_t, 3>& nSpacePoints);
		bool Compute() noexcept override;

		const auto& GetSpaceDiscretizations() const noexcept { return _spaceDiscretizations; }
		const auto& GetNumberOfSpacePoints() const noexcept { return _nSpacePoints; }
	private:
		const InputData<Real>& _inputData;
		const std::array<std::size_t, 3>& _nSpacePoints;

		using TridiagonalMatrices = std::vector<la::TridiagonalMatrix<Real>>;
		std::array<TridiagonalMatrices, 3> _spaceDiscretizations;
	};
}	 // namespace pde

extern template class pde::AdiSpaceDiscretizer<float>;
extern template class pde::AdiSpaceDiscretizer<double>;
