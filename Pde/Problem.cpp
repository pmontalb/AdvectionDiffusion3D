
#include "Problem.h"

#include <cassert>
#include <cmath>

namespace pde
{
	namespace detail
	{
		template<typename T>
		struct Tolerance;

		template<>
		struct Tolerance<float>
		{
			static constexpr float value = 1e-7f;
		};

		template<>
		struct Tolerance<double>
		{
			static constexpr double value = 1e-7;
		};
	}	 // namespace detail

	template<typename Real>
	Problem<Real>::Problem(const InputData<Real>& inputData, pde::Configuration configuration) noexcept
		: _inputData(inputData), _configuration(configuration), _solution(inputData.initialCondition)
	{
		assert(inputData.initialCondition.size() == _configuration.nSpacePoints[0] * _configuration.nSpacePoints[1] * _configuration.nSpacePoints[2]);
	}

	template<typename Real>
	void Problem<Real>::Advance() noexcept
	{
		const auto dx = _inputData.spaceGrids[0][1] - _inputData.spaceGrids[0][0];
		const auto dy = _inputData.spaceGrids[1][1] - _inputData.spaceGrids[1][0];
		const auto dz = _inputData.spaceGrids[2][1] - _inputData.spaceGrids[2][0];
		assert(std::isfinite(dx) && dx > detail::Tolerance<Real>::value);
		assert(std::isfinite(dy) && dy > detail::Tolerance<Real>::value);
		assert(std::isfinite(dz) && dz > detail::Tolerance<Real>::value);

		static constexpr auto one = Real(1.0);
		static constexpr auto two = Real(2.0);
		static constexpr auto half = Real(0.5);

		const auto& u = _inputData.velocityField[0];
		const auto& v = _inputData.velocityField[1];
		const auto& w = _inputData.velocityField[2];
		assert(u.size() == _configuration.nSpacePoints[0] * _configuration.nSpacePoints[1] * _configuration.nSpacePoints[2]);
		assert(v.size() == _configuration.nSpacePoints[0] * _configuration.nSpacePoints[1] * _configuration.nSpacePoints[2]);
		assert(w.size() == _configuration.nSpacePoints[0] * _configuration.nSpacePoints[1] * _configuration.nSpacePoints[2]);

		const auto dt = _inputData.deltaTime;
		assert(std::isfinite(dt) && dt > detail::Tolerance<Real>::value);

		const auto Kx = _inputData.diffusionCoefficients[0];
		const auto Ky = _inputData.diffusionCoefficients[1];
		const auto Kz = _inputData.diffusionCoefficients[2];
		assert(std::isfinite(Kx) && Kx >= Real(0.0));
		assert(std::isfinite(Ky) && Ky >= Real(0.0));
		assert(std::isfinite(Kz) && Kz >= Real(0.0));

		for (std::size_t k = 1; k < _configuration.nSpacePoints[2] - 1; ++k)
		{
			assert(std::abs(_inputData.spaceGrids[2][k + 1] - _inputData.spaceGrids[2][k] - dz) < detail::Tolerance<Real>::value);
			for (std::size_t i = 1; i < _configuration.nSpacePoints[0] - 1; ++i)
			{
				assert(std::abs(_inputData.spaceGrids[0][i + 1] - _inputData.spaceGrids[0][i] - dx) < detail::Tolerance<Real>::value);
				for (std::size_t j = 1; j < _configuration.nSpacePoints[1] - 1; ++j)
				{
					assert(std::abs(_inputData.spaceGrids[1][j + 1] - _inputData.spaceGrids[1][j] - dx) < detail::Tolerance<Real>::value);

					auto advectionX = (u[GetIndex(i + 1, j, k)] * _solution[GetIndex(i + 1, j, k)]);
					advectionX -= (u[GetIndex(i - 1, j, k)] * _solution[GetIndex(i - 1, j, k)]);
					advectionX /= two * dx;

					auto advectionY = (v[GetIndex(i, j, k)] * _solution[GetIndex(i, j + 1, k)]);
					advectionY -= (v[GetIndex(i, j - 1, k)] * _solution[GetIndex(i, j - 1, k)]);
					advectionY /= two * dy;

					auto advectionZ = (w[GetIndex(i, j, k + 1)] * _solution[GetIndex(i, j, k + 1)]);
					advectionZ -= (w[GetIndex(i, j, k - 1)] * _solution[GetIndex(i, j, k - 1)]);
					advectionZ /= two * dz;

					const auto advection = advectionX + advectionY + advectionZ;

					const auto diffusionX =
						Kx * (_solution[GetIndex(i + 1, j, k)] - two * _solution[GetIndex(i, j, k)] + _solution[GetIndex(i - 1, j, k)]) / (dx * dx);
					const auto diffusionY =
						Ky * (_solution[GetIndex(i, j + 1, k)] - two * _solution[GetIndex(i, j, k)] + _solution[GetIndex(i, j - 1, k)]) / (dy * dy);
					const auto diffusionZ =
						Kz * (_solution[GetIndex(i, j, k + 1)] - two * _solution[GetIndex(i, j, k)] + _solution[GetIndex(i, j, k - 1)]) / (dz * dz);

					const auto diffusion = diffusionX + diffusionY + diffusionZ;

					switch (_configuration.solverType)
					{
						case SolverType::ExplicitEuler:
							_solution[GetIndex(i, j, k)] -= dt * (advection + diffusion);
							break;
						case SolverType::LaxWendroff:
							{
								const auto velocityDiffusionX = (u[GetIndex(i + 1, j, k)] - two * u[GetIndex(i, j, k)] + u[GetIndex(i - 1, j, k)]) / (dx * dx);
								const auto velocityDiffusionY = (v[GetIndex(i, j + 1, k)] - two * v[GetIndex(i, j, k)] + v[GetIndex(i, j - 1, k)]) / (dy * dy);
								const auto velocityDiffusionZ = (w[GetIndex(i, j, k + 1)] - two * w[GetIndex(i, j, k)] + w[GetIndex(i, j, k - 1)]) / (dz * dz);

								const auto velocityGradientX = (u[GetIndex(i + 1, j, k)] - u[GetIndex(i - 1, j, k)]) / (two * dx);
								const auto velocityGradientY = (v[GetIndex(i, j + 1, k)] - v[GetIndex(i, j - 1, k)]) / (two * dy);
								const auto velocityGradientZ = (w[GetIndex(i, j, k + 1)] - w[GetIndex(i, j, k - 1)]) / (two * dz);
								const auto velocityGradient = velocityGradientX + velocityGradientY + velocityGradientZ;

								_solution[GetIndex(i, j, k)] -= dt * (advection + diffusion) * (one + half * dt * velocityGradient);
								_solution[GetIndex(i, j, k)] -= half * dt * dt * u[GetIndex(i, j, k)] * velocityDiffusionX;
								_solution[GetIndex(i, j, k)] -= half * dt * dt * v[GetIndex(i, j, k)] * velocityDiffusionY;
								_solution[GetIndex(i, j, k)] -= half * dt * dt * w[GetIndex(i, j, k)] * velocityDiffusionZ;
								break;
							}
						default:
							assert(false);
							break;
					}
				}
			}
		}

		SetZeroFluxBoundaryConditions();
	}

	template<typename Real>
	void Problem<Real>::SetZeroFluxBoundaryConditions()
	{
		// *** Boundary Conditions on X ***
		for (std::size_t k = 0; k < _configuration.nSpacePoints[2]; ++k)
		{
			for (std::size_t j = 0; j < _configuration.nSpacePoints[1]; ++j)
			{
				_solution[GetIndex(0, j, k)] = _solution[GetIndex(1, j, k)];
				_solution[GetIndex(_configuration.nSpacePoints[0] - 1, j, k)] = _solution[GetIndex(_configuration.nSpacePoints[0] - 2, j, k)];
			}
		}

		// *** Boundary Conditions on Y ***
		for (std::size_t k = 0; k < _configuration.nSpacePoints[2]; ++k)
		{
			for (std::size_t i = 0; i < _configuration.nSpacePoints[1]; ++i)
			{
				_solution[GetIndex(i, 0, k)] = _solution[GetIndex(i, 1, k)];
				_solution[GetIndex(i, _configuration.nSpacePoints[1] - 1, k)] = _solution[GetIndex(i, _configuration.nSpacePoints[1] - 2, k)];
			}
		}

		// *** Boundary Conditions on Z ***
		for (std::size_t i = 0; i < _configuration.nSpacePoints[0]; ++i)
		{
			for (std::size_t j = 0; j < _configuration.nSpacePoints[1]; ++j)
			{
				_solution[GetIndex(i, j, 0)] = _solution[GetIndex(i, j, 1)];
				_solution[GetIndex(i, j, _configuration.nSpacePoints[2] - 1)] = _solution[GetIndex(i, j, _configuration.nSpacePoints[2] - 2)];
			}
		}
	}
}	 // namespace pde

template class pde::Problem<float>;
template class pde::Problem<double>;
