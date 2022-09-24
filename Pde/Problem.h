
#pragma once

#include "Configuration.h"
#include "InputData.h"

namespace pde
{
	template<typename Real>
	class Problem
	{
	public:
		Problem(const InputData<Real>& inputData, Configuration configuration) noexcept;
		const auto& GetSolution() const noexcept { return _solution; }

		void Advance() noexcept;

	private:
		[[nodiscard]] constexpr size_t GetIndex(const size_t i, const size_t j, const size_t k) const noexcept
		{
			return _configuration.GetIndex(i, j, k);
		}

		void SetZeroFluxBoundaryConditions();

	private:
		const InputData<Real>& _inputData;
		Configuration _configuration;

		std::vector<Real> _solution;
	};
}	 // namespace pde

extern template class pde::Problem<float>;
extern template class pde::Problem<double>;
