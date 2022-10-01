
#include "TridiagonalMatrix.h"

#include "Pde/Tolerance.h"

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
		}

		void Solve(Eigen::VectorX<Real>& inOut)
		{
			_cache.resize(static_cast<int>(_matrix.Size()));
			Solve(inOut, _cache);
		}

		void Solve(Eigen::VectorX<Real>& inOut, Eigen::VectorX<Real>& cache)
		{
			assert(inOut.size() == static_cast<int>(_matrix.Size()));
			assert(inOut.size() == cache.size());

			static constexpr auto one = Real(1);

			assert(std::abs(_matrix[0].diag) > pde::Tolerance<Real>::value);
			const auto diagInverse = one / _matrix[0].diag;
			cache[0] = _matrix[0].super * diagInverse;
			inOut[0] *= diagInverse;

			for (size_t i = 1; i < _matrix.Size(); ++i)
			{
				assert(std::abs(_matrix[i].diag - _matrix[i].sub * cache[static_cast<int>(i) - 1]) >  pde::Tolerance<Real>::value);
				assert(std::abs(_matrix[i].diag) > std::abs(_matrix[i].sub) + std::abs(_matrix[i].super));
				const auto m = one / (_matrix[i].diag - _matrix[i].sub * cache[static_cast<int>(i) - 1]);
				cache[static_cast<int>(i)] = _matrix[i].super * m;
				inOut[static_cast<int>(i)] = (inOut[static_cast<int>(i)] - _matrix[i].sub * inOut[static_cast<int>(i) - 1]) * m;
			}

			for (size_t i = _matrix.Size() - 1; i != 0; --i)
				inOut[static_cast<int>(i) - 1] -= cache[static_cast<int>(i) - 1] * inOut[static_cast<int>(i)];
		}
	private:
		const TridiagonalMatrix<Real>& _matrix;
		Eigen::VectorX<Real> _cache;
	};
}
