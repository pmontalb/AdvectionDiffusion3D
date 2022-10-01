
#include "CrankNicolsonTimeDiscretizer.h"

namespace pde
{
	template<typename Real>
	bool CrankNicolsonTimeDiscretizer<Real>::Precompute() noexcept
	{
		// B = I - 0.5 * dt * L
		_leftOperator.resize(_spaceDiscretizer.GetOperator().rows(), _spaceDiscretizer.GetOperator().cols());
		_leftOperator.setIdentity();
		_leftOperator -= _spaceDiscretizer.GetOperator() * (Real(0.5) * _inputData.deltaTime);

		_solver.compute(_leftOperator);

		// A = I + 0.5 * dt * L
		_rightOperator.resize(_spaceDiscretizer.GetOperator().rows(), _spaceDiscretizer.GetOperator().cols());
		_rightOperator.setIdentity();
		_rightOperator += _spaceDiscretizer.GetOperator() * (Real(0.5) * _inputData.deltaTime);

		return _solver.info() == Eigen::Success;
	}

	template<typename Real>
	void CrankNicolsonTimeDiscretizer<Real>::Compute(Eigen::VectorX<Real>& out, const Eigen::VectorX<Real>& in) noexcept
	{
		_cacheProduct.noalias() = _rightOperator * in;
		out.noalias() = _solver.solve(_cacheProduct);
		assert(_solver.info() == Eigen::Success);
	}
	template<typename Real>
	void CrankNicolsonTimeDiscretizer<Real>::Compute(Eigen::VectorX<Real>& out, const Eigen::VectorX<Real>& in, const Eigen::VectorX<Real>& sourceTerm) noexcept
	{
		_cacheProduct.noalias() = _rightOperator * in + sourceTerm * _inputData.deltaTime * Real(0.5);
		out.noalias() = _solver.solve(_cacheProduct);
		assert(_solver.info() == Eigen::Success);
	}

}	 // namespace pde

template class pde::CrankNicolsonTimeDiscretizer<float>;
template class pde::CrankNicolsonTimeDiscretizer<double>;
