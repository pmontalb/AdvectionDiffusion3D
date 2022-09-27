
#include "ImplicitTimeDiscretizer.h"

namespace pde
{
	template<typename Real>
	bool ImplicitTimeDiscretizer<Real>::Precompute() noexcept
	{
		// B = I - dt * L
		_operator.resize(_spaceDiscretizer.GetOperator().rows(), _spaceDiscretizer.GetOperator().cols());
		_operator.setIdentity();
		_operator -= _spaceDiscretizer.GetOperator() * _inputData.deltaTime;

		_solver.compute(_operator);
		return _solver.info() == Eigen::Success;
	}

	template<typename Real>
	void ImplicitTimeDiscretizer<Real>::Compute(Eigen::VectorX<Real>& out, const Eigen::VectorX<Real>& in) noexcept
	{
		_cache.noalias() = _solver.solve(in);
		assert(_sparseSolver.info() == Eigen::Success);
		out = _cache;
	}

}	 // namespace pde

template class pde::ImplicitTimeDiscretizer<float>;
template class pde::ImplicitTimeDiscretizer<double>;
