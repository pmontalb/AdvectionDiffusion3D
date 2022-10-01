
#include "AdiSpaceDiscretizer.h"
#include "Pde/Tolerance.h"

namespace pde
{
	template<typename Real>
	AdiSpaceDiscretizer<Real>::AdiSpaceDiscretizer(const InputData<Real>& inputData, const std::array<std::size_t, 3>& nSpacePoints) : _inputData(inputData), _nSpacePoints(nSpacePoints)
	{
		_spaceDiscretizations[0].resize(_nSpacePoints[1] * _nSpacePoints[2]);
		_spaceDiscretizations[1].resize(_nSpacePoints[0] * _nSpacePoints[2]);
		_spaceDiscretizations[2].resize(_nSpacePoints[0] * _nSpacePoints[1]);

		for (size_t i = 0; i < _spaceDiscretizations.size(); ++i)
		{
			for (auto& matrix : _spaceDiscretizations[i])
			{
				matrix.Resize(_nSpacePoints[i]);
				matrix.Front() = { 0.0, 0.0, 0.0 };
				matrix.Back() = { 0.0, 0.0, 0.0 };
			}
		}
	}

	template<typename Real>
	bool AdiSpaceDiscretizer<Real>::Compute() noexcept
	{
		const auto dx = _inputData.spaceGrids[0][1] - _inputData.spaceGrids[0][0];
		const auto dy = _inputData.spaceGrids[1][1] - _inputData.spaceGrids[1][0];
		const auto dz = _inputData.spaceGrids[2][1] - _inputData.spaceGrids[2][0];
		assert(std::isfinite(dx) && dx > Tolerance<Real>::value);
		assert(std::isfinite(dy) && dy > Tolerance<Real>::value);
		assert(std::isfinite(dz) && dz > Tolerance<Real>::value);

		static constexpr auto two = Real(2.0);

		const auto& u = _inputData.velocityField[0];
		const auto& v = _inputData.velocityField[1];
		const auto& w = _inputData.velocityField[2];
		assert(u.size() == _nSpacePoints[0] * _nSpacePoints[1] * _nSpacePoints[2]);
		assert(v.size() == _nSpacePoints[0] * _nSpacePoints[1] * _nSpacePoints[2]);
		assert(w.size() == _nSpacePoints[0] * _nSpacePoints[1] * _nSpacePoints[2]);

		assert(std::isfinite(_inputData.deltaTime) && _inputData.deltaTime > Tolerance<Real>::value);

		const auto Kx = _inputData.diffusionCoefficients[0];
		const auto Ky = _inputData.diffusionCoefficients[1];
		const auto Kz = _inputData.diffusionCoefficients[2];
		assert(std::isfinite(Kx) && Kx >= Real(0.0));
		assert(std::isfinite(Ky) && Ky >= Real(0.0));
		assert(std::isfinite(Kz) && Kz >= Real(0.0));

		// discretize each dimension separately
		for (std::size_t i = 0; i < _nSpacePoints[0] - 0; ++i)
		{
			for (std::size_t j = 0; j < _nSpacePoints[1] - 0; ++j)
			{
				for (std::size_t k = 0; k < _nSpacePoints[2] - 0; ++k)
				{
					// x
					auto& xDiscretizer = _spaceDiscretizations[0][j + k * _nSpacePoints[1]][i];

					if (i > 0 && i < _nSpacePoints[0] - 1)
					{
						const auto indexPlusX = GetIndex(i + 1, j, k, _nSpacePoints);
						const auto indexMinusX = GetIndex(i - 1, j, k, _nSpacePoints);

						xDiscretizer.sub = -u[indexMinusX] / (two * dx);
						xDiscretizer.sub += Kx / (dx * dx);

						xDiscretizer.diag = -two * Kx / (dx * dx);

						xDiscretizer.super = u[indexPlusX] / (two * dx);
						xDiscretizer.super += Kx / (dx * dx);
					}

					// y
					auto& yDiscretizer = _spaceDiscretizations[1][i + k * _nSpacePoints[0]][j];

					if (j > 0 && j < _nSpacePoints[1] - 1)
					{
						const auto indexPlusY = GetIndex(i, j + 1, k, _nSpacePoints);
						const auto indexMinusY = GetIndex(i, j - 1, k, _nSpacePoints);

						yDiscretizer.sub = -v[indexMinusY] / (two * dy);
						yDiscretizer.sub += Ky / (dy * dy);

						yDiscretizer.diag = -two * Ky / (dy * dy);

						yDiscretizer.super = v[indexPlusY] / (two * dy);
						yDiscretizer.super += Ky / (dy * dy);
					}

					// z
					auto& zDiscretizer = _spaceDiscretizations[2][j + i * _nSpacePoints[1]][k];

					if (k > 0 && k < _nSpacePoints[2] - 1)
					{
						const auto indexPlusZ = GetIndex(i, j, k + 1, _nSpacePoints);
						const auto indexMinusZ = GetIndex(i, j, k - 1, _nSpacePoints);

						zDiscretizer.sub = -w[indexMinusZ] / (two * dz);
						zDiscretizer.sub += Kz / (dz * dz);

						zDiscretizer.diag = -two * Kz / (dz * dz);

						zDiscretizer.super = w[indexPlusZ] / (two * dz);
						zDiscretizer.super += Kz / (dz * dz);
					}
				}
			}
		}

		return true;
	}

}	 // namespace pde

template class pde::AdiSpaceDiscretizer<float>;
template class pde::AdiSpaceDiscretizer<double>;
