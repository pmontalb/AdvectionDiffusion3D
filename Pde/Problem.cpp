
#include "Problem.h"

#include "Discretizers/CenteredDifferenceSpaceDiscretizer.h"
#include "Discretizers/AdiSpaceDiscretizer.h"

#include "Discretizers/ExplicitTimeDiscretizer.h"
#include "Discretizers/ImplicitTimeDiscretizer.h"
#include "Discretizers/CrankNicolsonTimeDiscretizer.h"
#include "Discretizers/AdiTimeDiscretizer.h"

#include "Pde/Tolerance.h"

#include <Eigen/Sparse>

#include <cassert>
#include <cmath>
#include <iostream>

namespace pde
{
	template<typename Real>
	Problem<Real>::Problem(const InputData<Real>& inputData) noexcept : _inputData(inputData)
	{
		_solution.resize(static_cast<long>(_inputData.initialCondition.size()));
		for (size_t i = 0; i < static_cast<size_t>(_solution.size()); ++i)
			_solution[i] = _inputData.initialCondition[i];
		for (size_t n = 0; n < _inputData.spaceGrids.size(); ++n)
			_nSpacePoints[n] = _inputData.spaceGrids[n].size();
		assert(inputData.initialCondition.size() == _nSpacePoints[0] * _nSpacePoints[1] * _nSpacePoints[2]);
	}

	template<typename Real>
	void Problem<Real>::Advance(const SolverType solverType) noexcept
	{
		AdvanceNoBoundaryConditions(solverType);
		SetZeroFluxBoundaryConditions();
	}

	template<typename Real>
	void Problem<Real>::Advance(const SolverType solverType, const Eigen::VectorX<Real>& sourceTerm) noexcept
	{
		Advance(solverType);
		_solution.noalias() += sourceTerm * _inputData.deltaTime;
	}

	template<typename Real>
	void Problem<Real>::AdvanceNoBoundaryConditions(const SolverType solverType) noexcept
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

		const auto dt = _inputData.deltaTime;
		assert(std::isfinite(dt) && dt > Tolerance<Real>::value);

		const auto Kx = _inputData.diffusionCoefficients[0];
		const auto Ky = _inputData.diffusionCoefficients[1];
		const auto Kz = _inputData.diffusionCoefficients[2];
		assert(std::isfinite(Kx) && Kx >= Real(0.0));
		assert(std::isfinite(Ky) && Ky >= Real(0.0));
		assert(std::isfinite(Kz) && Kz >= Real(0.0));

		for (std::size_t k = 1; k < _nSpacePoints[2] - 1; ++k)
		{
			assert(std::abs(_inputData.spaceGrids[2][k + 1] - _inputData.spaceGrids[2][k] - dz) < Real(2) * Tolerance<Real>::value);
			for (std::size_t i = 1; i < _nSpacePoints[0] - 1; ++i)
			{
				assert(std::abs(_inputData.spaceGrids[0][i + 1] - _inputData.spaceGrids[0][i] - dx) < Real(2) * Tolerance<Real>::value);
				for (std::size_t j = 1; j < _nSpacePoints[1] - 1; ++j)
				{
					assert(std::abs(_inputData.spaceGrids[1][j + 1] - _inputData.spaceGrids[1][j] - dy) < Real(2) * Tolerance<Real>::value);

					auto advectionX = (u[GetIndex(i + 1, j, k)] * _solution[GetIndex(i + 1, j, k)]);
					advectionX -= (u[GetIndex(i - 1, j, k)] * _solution[GetIndex(i - 1, j, k)]);
					advectionX /= two * dx;

					auto advectionY = (v[GetIndex(i, j + 1, k)] * _solution[GetIndex(i, j + 1, k)]);
					advectionY -= (v[GetIndex(i, j - 1, k)] * _solution[GetIndex(i, j - 1, k)]);
					advectionY /= two * dy;

					auto advectionZ = (w[GetIndex(i, j, k + 1)] * _solution[GetIndex(i, j, k + 1)]);
					advectionZ -= (w[GetIndex(i, j, k - 1)] * _solution[GetIndex(i, j, k - 1)]);
					advectionZ /= two * dz;

					const auto advection = advectionX + advectionY + advectionZ;

					const auto diffusionX = Kx * (_solution[GetIndex(i + 1, j, k)] - two * _solution[GetIndex(i, j, k)] + _solution[GetIndex(i - 1, j, k)]) / (dx * dx);
					const auto diffusionY = Ky * (_solution[GetIndex(i, j + 1, k)] - two * _solution[GetIndex(i, j, k)] + _solution[GetIndex(i, j - 1, k)]) / (dy * dy);
					const auto diffusionZ = Kz * (_solution[GetIndex(i, j, k + 1)] - two * _solution[GetIndex(i, j, k)] + _solution[GetIndex(i, j, k - 1)]) / (dz * dz);

					const auto diffusion = diffusionX + diffusionY + diffusionZ;

					switch (solverType)
					{
						case SolverType::ExplicitEuler:
							_solution[GetIndex(i, j, k)] += dt * (advection + diffusion);
							break;
						case SolverType::LaxWendroff:
							{
								const auto crossDiffusionX = (u[GetIndex(i + 1, j, k)] * _solution[GetIndex(i + 1, j, k)] - two * u[GetIndex(i, j, k)] * _solution[GetIndex(i, j, k)] +
															  u[GetIndex(i - 1, j, k)] * _solution[GetIndex(i - 1, j, k)]) /
															 (dx * dx);
								const auto crossDiffusionY = (v[GetIndex(i, j + 1, k)] * _solution[GetIndex(i, j + 1, k)] - two * v[GetIndex(i, j, k)] * _solution[GetIndex(i, j, k)] +
															  v[GetIndex(i, j - 1, k)] * _solution[GetIndex(i, j - 1, k)]) /
															 (dy * dy);
								const auto crossDiffusionZ = (w[GetIndex(i, j, k + 1)] * _solution[GetIndex(i, j, k + 1)] - two * w[GetIndex(i, j, k)] * _solution[GetIndex(i, j, k)] +
															  w[GetIndex(i, j, k - 1)] * _solution[GetIndex(i, j, k - 1)]) /
															 (dz * dz);

								const auto velocityGradientX = (u[GetIndex(i + 1, j, k)] - u[GetIndex(i - 1, j, k)]) / (two * dx);
								const auto velocityGradientY = (v[GetIndex(i, j + 1, k)] - v[GetIndex(i, j - 1, k)]) / (two * dy);
								const auto velocityGradientZ = (w[GetIndex(i, j, k + 1)] - w[GetIndex(i, j, k - 1)]) / (two * dz);
								const auto velocityGradient = velocityGradientX + velocityGradientY + velocityGradientZ;

								_solution[GetIndex(i, j, k)] += dt * (advection + diffusion) * (one + half * dt * velocityGradient);
								_solution[GetIndex(i, j, k)] += half * dt * dt * u[GetIndex(i, j, k)] * crossDiffusionX;
								_solution[GetIndex(i, j, k)] += half * dt * dt * v[GetIndex(i, j, k)] * crossDiffusionY;
								_solution[GetIndex(i, j, k)] += half * dt * dt * w[GetIndex(i, j, k)] * crossDiffusionZ;
								break;
							}
						default:
							assert(false);
							break;
					}
				}
			}
		}
	}
	template<typename Real>
	void Problem<Real>::AdvanceNoBoundaryConditions(const SolverType solverType, const Eigen::VectorX<Real>& sourceTerm) noexcept
	{
		AdvanceNoBoundaryConditions(solverType);
		_solution.noalias() += sourceTerm * _inputData.deltaTime;
	}

	template<typename Real>
	void Problem<Real>::SetZeroFluxBoundaryConditions()
	{
		// *** Boundary Conditions on X ***
		for (std::size_t k = 0; k < _nSpacePoints[2]; ++k)
		{
			for (std::size_t j = 0; j < _nSpacePoints[1]; ++j)
			{
				_solution[GetIndex(0, j, k)] = _solution[GetIndex(1, j, k)];
				_solution[GetIndex(_nSpacePoints[0] - 1, j, k)] = _solution[GetIndex(_nSpacePoints[0] - 2, j, k)];
			}
		}

		// *** Boundary Conditions on Y ***
		for (std::size_t k = 0; k < _nSpacePoints[2]; ++k)
		{
			for (std::size_t i = 0; i < _nSpacePoints[1]; ++i)
			{
				_solution[GetIndex(i, 0, k)] = _solution[GetIndex(i, 1, k)];
				_solution[GetIndex(i, _nSpacePoints[1] - 1, k)] = _solution[GetIndex(i, _nSpacePoints[1] - 2, k)];
			}
		}

		// *** Boundary Conditions on Z ***
		for (std::size_t i = 0; i < _nSpacePoints[0]; ++i)
		{
			for (std::size_t j = 0; j < _nSpacePoints[1]; ++j)
			{
				_solution[GetIndex(i, j, 0)] = _solution[GetIndex(i, j, 1)];
				_solution[GetIndex(i, j, _nSpacePoints[2] - 1)] = _solution[GetIndex(i, j, _nSpacePoints[2] - 2)];
			}
		}

		SetZeroFluxBoundaryConditionsAtObstacles();
	}

	template<typename Real>
	void Problem<Real>::SetZeroFluxBoundaryConditionsAtObstacles()
	{
		// TODO
	}

	template<typename Real>
	void LinearOperatorProblem<Real>::Precompute(const SolverType solverType) noexcept
	{
		if (_lastSolverType == SolverType::Null || _lastSolverType != solverType)
		{
			MakeSpaceOperator(solverType);

			if (!_timeDiscretizer || _lastSolverType != solverType)
			{
				switch (solverType)
				{
					case SolverType::ExplicitEuler:
					case SolverType::LaxWendroff:
						_timeDiscretizer = std::make_unique<ExplicitTimeDiscretizer<Real>>(_inputData, static_cast<CenteredDifferenceSpaceDiscretizer<Real>&>(*_spaceDiscretizer));
						break;
					case SolverType::ImplicitEuler:
						_timeDiscretizer = std::make_unique<ImplicitTimeDiscretizer<Real>>(_inputData, static_cast<CenteredDifferenceSpaceDiscretizer<Real>&>(*_spaceDiscretizer));
						break;
					case SolverType::CrankNicolson:
						_timeDiscretizer = std::make_unique<CrankNicolsonTimeDiscretizer<Real>>(_inputData, static_cast<CenteredDifferenceSpaceDiscretizer<Real>&>(*_spaceDiscretizer));
						break;
					case SolverType::ADI:
						_timeDiscretizer = std::make_unique<AdiTimeDiscretizer<Real>>(_inputData, static_cast<AdiSpaceDiscretizer<Real>&>(*_spaceDiscretizer));
						break;

					default:
						_timeDiscretizer.reset();
						assert(false);
						break;
				}
			}
#ifndef NDEBUG
			const auto success = _timeDiscretizer->Precompute();
			assert(success);
#else
			_timeDiscretizer->Precompute();
#endif

			_lastSolverType = solverType;
			_cache.resize(_solution.size());
		}
	}

	template<typename Real>
	void LinearOperatorProblem<Real>::MakeSpaceOperator(const SolverType solverType) noexcept
	{
		if (!_spaceDiscretizer || _lastSolverType != solverType)
		{
			switch (solverType)
			{
				case SolverType::ExplicitEuler:
				case SolverType::LaxWendroff:
				case SolverType::ImplicitEuler:
				case SolverType::CrankNicolson:
					_spaceDiscretizer = std::make_unique<CenteredDifferenceSpaceDiscretizer<Real>>(_inputData, _nSpacePoints, solverType);
					break;
				case SolverType::ADI:
					_spaceDiscretizer = std::make_unique<AdiSpaceDiscretizer<Real>>(_inputData, _nSpacePoints);
					break;
				default:
					_spaceDiscretizer.reset();
					assert(false);
					break;
			}
		}
#ifndef NDEBUG
		const auto success = _spaceDiscretizer->Compute();
		assert(success);
#else
		_spaceDiscretizer->Compute();
#endif
	}

	template<typename Real>
	void LinearOperatorProblem<Real>::Advance(const SolverType solverType) noexcept
	{
		AdvanceNoBoundaryConditions(solverType);
		this->SetZeroFluxBoundaryConditions();
	}

	template<typename Real>
	void LinearOperatorProblem<Real>::Advance(const SolverType solverType, const Eigen::VectorX<Real>& sourceTerm) noexcept
	{
		AdvanceNoBoundaryConditions(solverType, sourceTerm);
		this->SetZeroFluxBoundaryConditions();
	}

	template<typename Real>
	void LinearOperatorProblem<Real>::AdvanceNoBoundaryConditions(const SolverType solverType) noexcept
	{
		Precompute(solverType);
		_timeDiscretizer->Compute(_cache, _solution);
		_solution.noalias() = _cache;
	}
	template<typename Real>
	void LinearOperatorProblem<Real>::AdvanceNoBoundaryConditions(const SolverType solverType, const Eigen::VectorX<Real>& sourceTerm) noexcept
	{
		Precompute(solverType);
		_timeDiscretizer->Compute(_cache, _solution, sourceTerm);
		_solution.noalias() = _cache;
	}

}	 // namespace pde

template class pde::Problem<float>;
template class pde::Problem<double>;

template class pde::LinearOperatorProblem<float>;
template class pde::LinearOperatorProblem<double>;
