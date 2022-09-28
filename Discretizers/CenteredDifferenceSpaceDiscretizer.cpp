
#include "CenteredDifferenceSpaceDiscretizer.h"
#include "Pde/Tolerance.h"

namespace pde
{
	template<typename Real>
	bool CenteredDifferenceSpaceDiscretizer<Real>::Compute() noexcept
	{
		const auto dx = _inputData.spaceGrids[0][1] - _inputData.spaceGrids[0][0];
		const auto dy = _inputData.spaceGrids[1][1] - _inputData.spaceGrids[1][0];
		const auto dz = _inputData.spaceGrids[2][1] - _inputData.spaceGrids[2][0];
		assert(std::isfinite(dx) && dx > Tolerance<Real>::value);
		assert(std::isfinite(dy) && dy > Tolerance<Real>::value);
		assert(std::isfinite(dz) && dz > Tolerance<Real>::value);

		static constexpr auto one = Real(1.0);
		static constexpr auto two = Real(2.0);
		static constexpr auto half = Real(0.5);

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

		const auto totalSize = static_cast<long>(_nSpacePoints[0] * _nSpacePoints[1] * _nSpacePoints[2]);

		// L
		_operator.resize(totalSize, totalSize);

		// build the triplets
		std::vector<Eigen::Triplet<Real>> nonZeroElements;
		nonZeroElements.reserve(7 * static_cast<size_t>(totalSize));
		for (std::size_t k = 1; k < _nSpacePoints[2] - 1; ++k)
		{
			for (std::size_t i = 1; i < _nSpacePoints[0] - 1; ++i)
			{
				for (std::size_t j = 1; j < _nSpacePoints[1] - 1; ++j)
				{
					const auto index = GetIndex(i, j, k, _nSpacePoints);

					const auto indexPlusX = GetIndex(i + 1, j, k, _nSpacePoints);
					const auto indexMinusX = GetIndex(i - 1, j, k, _nSpacePoints);

					const auto indexPlusY = GetIndex(i, j + 1, k, _nSpacePoints);
					const auto indexMinusY = GetIndex(i, j - 1, k, _nSpacePoints);

					const auto indexPlusZ = GetIndex(i, j, k + 1, _nSpacePoints);
					const auto indexMinusZ = GetIndex(i, j, k - 1, _nSpacePoints);

					Real adjustingFactor = one;
					if (_solverType == SolverType::LaxWendroff)
					{
						const auto velocityGradientX = (u[indexPlusX] - u[indexMinusX]) / (two * dx);
						const auto velocityGradientY = (v[indexPlusY] - v[indexMinusY]) / (two * dy);
						const auto velocityGradientZ = (w[indexPlusZ] - w[indexMinusZ]) / (two * dz);
						const auto velocityGradient = velocityGradientX + velocityGradientY + velocityGradientZ;
						adjustingFactor = one + half * _inputData.deltaTime * velocityGradient;
					}

					auto diagonal = -two * (Kx / (dx * dx) + Ky / (dy * dy) + Kz / (dz * dz));
					if (_solverType == SolverType::LaxWendroff)
					{
						diagonal *= adjustingFactor;

						diagonal += half * _inputData.deltaTime * u[index] * u[index] * (-two / (dx * dx));
						diagonal += half * _inputData.deltaTime * v[index] * v[index] * (-two / (dy * dy));
						diagonal += half * _inputData.deltaTime * w[index] * w[index] * (-two / (dz * dz));
					}
					nonZeroElements.emplace_back(index, index, diagonal);

					auto upDiagonalX = u[indexPlusX] / (two * dx);
					upDiagonalX += Kx / (dx * dx);
					if (_solverType == SolverType::LaxWendroff)
					{
						upDiagonalX *= adjustingFactor;
						upDiagonalX += half * _inputData.deltaTime * u[indexPlusX] * (u[indexPlusX] / (dx * dx));
					}
					nonZeroElements.emplace_back(index, indexPlusX, upDiagonalX);

					auto downDiagonalX = -u[indexMinusX] / (two * dx);
					downDiagonalX += Kx / (dx * dx);
					if (_solverType == SolverType::LaxWendroff)
					{
						downDiagonalX *= adjustingFactor;
						downDiagonalX += half * _inputData.deltaTime * u[indexMinusX] * (u[indexMinusX] / (dx * dx));
					}
					nonZeroElements.emplace_back(index, indexMinusX, downDiagonalX);

					auto upDiagonalY = v[indexPlusY] / (two * dy);
					upDiagonalY += Ky / (dy * dy);
					if (_solverType == SolverType::LaxWendroff)
					{
						upDiagonalY *= adjustingFactor;
						upDiagonalY += half * _inputData.deltaTime * v[indexPlusY] * (v[indexPlusY] / (dy * dy));
					}
					nonZeroElements.emplace_back(index, indexPlusY, upDiagonalY);

					auto downDiagonalY = -v[indexMinusY] / (two * dy);
					downDiagonalY += Ky / (dy * dy);
					if (_solverType == SolverType::LaxWendroff)
					{
						downDiagonalY *= adjustingFactor;
						downDiagonalY += half * _inputData.deltaTime * v[indexMinusY] * (v[indexMinusY] / (dy * dy));
					}
					nonZeroElements.emplace_back(index, indexMinusY, downDiagonalY);

					auto upDiagonalZ = w[indexPlusZ] / (two * dz);
					upDiagonalZ += Kz / (dz * dz);
					if (_solverType == SolverType::LaxWendroff)
					{
						upDiagonalZ *= adjustingFactor;
						upDiagonalZ += half * _inputData.deltaTime * w[indexPlusZ] * (w[indexPlusZ] / (dz * dz));
					}
					nonZeroElements.emplace_back(index, indexPlusZ, upDiagonalZ);

					auto downDiagonalZ = -w[indexMinusZ] / (two * dz);
					downDiagonalZ += Kz / (dz * dz);
					if (_solverType == SolverType::LaxWendroff)
					{
						downDiagonalZ *= adjustingFactor;
						downDiagonalZ += half * _inputData.deltaTime * w[indexMinusZ] * (w[indexMinusZ] / (dz * dz));
					}
					nonZeroElements.emplace_back(index, indexMinusZ, downDiagonalZ);
					assert(nonZeroElements.size() <= 7 * static_cast<size_t>(totalSize));
				}
			}
		}
		_operator.setFromTriplets(nonZeroElements.begin(), nonZeroElements.end());

		return true;
	}

}	 // namespace pde

template class pde::CenteredDifferenceSpaceDiscretizer<float>;
template class pde::CenteredDifferenceSpaceDiscretizer<double>;
