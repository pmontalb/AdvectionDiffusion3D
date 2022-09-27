
#include "TridiagonalMatrix.h"

#include <Eigen/Eigen>

namespace la
{
	template<typename Real>
	class TridiagonalSolver
	{
	public:
		explicit TridiagonalSolver(const TridiagonalMatrix<Real>& matrix) noexcept
			: _matrix(matrix)
		{
			_cache.resize(static_cast<int>(matrix.Size()));
		}

		void Solve(Eigen::VectorX<Real>& inOut)
		{
			assert(inOut.size() == static_cast<int>(_matrix.Size()));

			_cache[0] = _matrix[0].super / _matrix[0].diag;
			inOut[0] /= _matrix[0].diag;

			for (size_t i = 1; i < _matrix.Size(); ++i)
			{
				static constexpr auto one = Real(1);
				const auto m = one / (_matrix[i].diag - _matrix[i].sub * _cache[static_cast<int>(i) - 1]);
				_cache[static_cast<int>(i)] = _matrix[i].super * m;
				inOut[static_cast<int>(i)] = (inOut[static_cast<int>(i)] - _matrix[i].sub * inOut[static_cast<int>(i) - 1]) * m;
			}

			for (size_t i = _matrix.Size() - 1; i != 0; --i)
				inOut[static_cast<int>(i) - 1] -= _cache[static_cast<int>(i) - 1] * inOut[static_cast<int>(i)];
		}

	private:
		const TridiagonalMatrix<Real>& _matrix;
		Eigen::VectorX<Real> _cache;
	};
}
