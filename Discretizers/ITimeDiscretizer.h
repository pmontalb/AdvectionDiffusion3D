
#pragma once

#include <Eigen/Eigen>

namespace pde
{
	template<typename Real>
	class ITimeDiscretizer
	{
	public:
		virtual ~ITimeDiscretizer() = default;

		virtual bool Precompute() noexcept = 0;
		virtual void Compute(Eigen::VectorX<Real>& out, const Eigen::VectorX<Real>& in) noexcept = 0;
	};
}	 // namespace pde
