
#include "ExplicitTimeDiscretizer.h"

namespace pde
{
	template<typename Real>
	bool ExplicitTimeDiscretizer<Real>::Precompute() noexcept
	{
		// A = I + dt * L
		_operator.resize(_spaceDiscretizer.GetOperator().rows(), _spaceDiscretizer.GetOperator().cols());
		_operator.setIdentity();
		_operator += _spaceDiscretizer.GetOperator() * _inputData.deltaTime;

		return true;
	}

	template<typename Real>
	void ExplicitTimeDiscretizer<Real>::Compute(Eigen::VectorX<Real>& out, const Eigen::VectorX<Real>& in) noexcept
	{
		out.noalias() = _operator * in;
	}

}	 // namespace pde

template class pde::ExplicitTimeDiscretizer<float>;
template class pde::ExplicitTimeDiscretizer<double>;
