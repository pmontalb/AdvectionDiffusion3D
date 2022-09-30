
#include "AdiTimeDiscretizer.h"


namespace pde
{
	template<typename Real>
	AdiTimeDiscretizer<Real>::AdiTimeDiscretizer(const InputData<Real>& inputData, const AdiSpaceDiscretizer<Real>& spaceDiscretizer) : _inputData(inputData), _spaceDiscretizer(spaceDiscretizer)
	{
		const auto& spaceDiscretizations = _spaceDiscretizer.GetSpaceDiscretizations();

		const auto worker = [&](auto& operators)
		{
			for (size_t i = 0; i < operators.size(); ++i)
				operators[i].resize(spaceDiscretizations[i].size());
			for (size_t i = 0; i < operators.size(); ++i)
			{
				for (size_t j = 0; j < operators[i].size(); ++j)
				{
					auto& matrix = operators[i][j];
					matrix.Resize(spaceDiscretizations[i][0].Size());
				}
			}
		};
		worker(_leftOperators);
		worker(_rightOperators);

		for (size_t i = 0; i < _solverCache.size(); ++i)
		{
			const auto nPoints = static_cast<long>(_spaceDiscretizer.GetNumberOfSpacePoints()[i]);

			_solverCache[i].resize(nPoints);
			_dotProductCache[i].resize(nPoints);
			_inputCache[i].resize(nPoints);
		}

		_outCacheX.resize(static_cast<long>(_inputData.initialCondition.size()));
		_outCacheY.resize(static_cast<long>(_inputData.initialCondition.size()));
	}

	template<typename Real>
	bool AdiTimeDiscretizer<Real>::Precompute() noexcept
	{
		const auto& nSpacePoints = _spaceDiscretizer.GetNumberOfSpacePoints();
		const auto& spaceOperators = _spaceDiscretizer.GetSpaceDiscretizations();

		static constexpr auto one = Real(1.0);
		static constexpr auto half = Real(0.5);

		for (std::size_t k = 0; k < nSpacePoints[2]; ++k)
		{
			for (std::size_t i = 0; i < nSpacePoints[0]; ++i)
			{
				for (std::size_t j = 0; j < nSpacePoints[1]; ++j)
				{
					auto& spaceOperatorX = spaceOperators[0][j + k * nSpacePoints[1]][i];
					auto& spaceOperatorY = spaceOperators[1][i + k * nSpacePoints[0]][j];
					auto& spaceOperatorZ = spaceOperators[2][j + i * nSpacePoints[1]][k];

					auto& leftOperatorX = _leftOperators[0][j + k * nSpacePoints[1]][i];
					auto& leftOperatorY = _leftOperators[1][i + k * nSpacePoints[0]][j];
					auto& leftOperatorZ = _leftOperators[2][j + i * nSpacePoints[1]][k];

					auto& rightOperatorX = _rightOperators[0][j + k * nSpacePoints[1]][i];
					auto& rightOperatorY = _rightOperators[1][i + k * nSpacePoints[0]][j];
					auto& rightOperatorZ = _rightOperators[2][j + i * nSpacePoints[1]][k];

					// B_{x|y|z} = I - 0.5 * dt * L_{x|y|z}
					leftOperatorX = { -half * _inputData.deltaTime * spaceOperatorX.sub, one - half * _inputData.deltaTime * spaceOperatorX.diag, -half * _inputData.deltaTime * spaceOperatorX.super };
					leftOperatorY = { -half * _inputData.deltaTime * spaceOperatorY.sub, one - half * _inputData.deltaTime * spaceOperatorY.diag, -half * _inputData.deltaTime * spaceOperatorY.super };
					leftOperatorZ = { -half * _inputData.deltaTime * spaceOperatorZ.sub, one - half * _inputData.deltaTime * spaceOperatorZ.diag, -half * _inputData.deltaTime * spaceOperatorZ.super };

					// A_{x|y|z} = I + 0.5 * dt * L_{x|y|z}
					rightOperatorX = {
						half * _inputData.deltaTime * spaceOperatorX.sub,
						one + half * _inputData.deltaTime * spaceOperatorX.diag,
						half * _inputData.deltaTime * spaceOperatorX.super,
					};
					rightOperatorY = {
						half * _inputData.deltaTime * spaceOperatorY.sub,
						one + half * _inputData.deltaTime * spaceOperatorY.diag,
						half * _inputData.deltaTime * spaceOperatorY.super,
					};
					rightOperatorZ = {
						half * _inputData.deltaTime * spaceOperatorZ.sub,
						one + half * _inputData.deltaTime * spaceOperatorZ.diag,
						half * _inputData.deltaTime * spaceOperatorZ.super,
					};
				}
			}
		}

		return true;
	}

	template<typename Real>
	void AdiTimeDiscretizer<Real>::Compute(Eigen::VectorX<Real>& out, const Eigen::VectorX<Real>& in) noexcept
	{
		const auto& nSpacePoints = _spaceDiscretizer.GetNumberOfSpacePoints();
		const auto& spaceOperators = _spaceDiscretizer.GetSpaceDiscretizations();

		// x-solve
		for (std::size_t k = 0; k < nSpacePoints[2]; ++k)
		{
			for (std::size_t j = 0; j < nSpacePoints[1]; ++j)
			{
				auto& leftOperatorX = _leftOperators[0][j + k * nSpacePoints[1]];
				auto& rightOperatorX = _rightOperators[0][j + k * nSpacePoints[1]];

				// extract inX = in[k, j, :] (this is C_{n - 1})
				for (size_t i = 0; i < nSpacePoints[0]; ++i)
					_inputCache[0][static_cast<int>(i)] = in[static_cast<int>(GetIndex(i, j, k, nSpacePoints))];

				// tmp = (I + 0.5 * dt * Lx) * C_{n - 1}
				rightOperatorX.Dot(_dotProductCache[0], _inputCache[0]);

				// tmp += dt * (Ly + Lz) * C_{n - 1}
				for (size_t i = 0; i < nSpacePoints[0]; ++i)
				{
					const auto& Ly = spaceOperators[1][i + k * nSpacePoints[0]][j];
					if (j > 0 && j < nSpacePoints[1] - 1)
					{
						_dotProductCache[0][static_cast<int>(i)] += _inputData.deltaTime * Ly.super * in[static_cast<int>(GetIndex(i, j + 1, k, nSpacePoints))];
						_dotProductCache[0][static_cast<int>(i)] += _inputData.deltaTime * Ly.diag * in[static_cast<int>(GetIndex(i, j, k, nSpacePoints))];
						_dotProductCache[0][static_cast<int>(i)] += _inputData.deltaTime * Ly.sub * in[static_cast<int>(GetIndex(i, j - 1, k, nSpacePoints))];
					}

					const auto& Lz = spaceOperators[2][j + i * nSpacePoints[1]][k];
					if (k > 0 && k < nSpacePoints[2] - 1)
					{
						_dotProductCache[0][static_cast<int>(i)] += _inputData.deltaTime * Lz.super * in[static_cast<int>(GetIndex(i, j, k + 1, nSpacePoints))];
						_dotProductCache[0][static_cast<int>(i)] += _inputData.deltaTime * Lz.diag * in[static_cast<int>(GetIndex(i, j, k, nSpacePoints))];
						_dotProductCache[0][static_cast<int>(i)] += _inputData.deltaTime * Lz.sub * in[static_cast<int>(GetIndex(i, j, k - 1, nSpacePoints))];
					}
				}

				// C_X = (I - 0.5 * dt * L_x) \ tmp
				la::TridiagonalSolver<Real> solver(leftOperatorX);
				solver.Solve(_dotProductCache[0], _solverCache[0]);

				// assign to the current output
				for (size_t i = 0; i < nSpacePoints[0]; ++i)
					_outCacheX[static_cast<int>(GetIndex(i, j, k, nSpacePoints))] = _dotProductCache[0][static_cast<int>(i)];
			}
		}

		// y-solve
		for (std::size_t k = 0; k < nSpacePoints[2]; ++k)
		{
			for (std::size_t i = 0; i < nSpacePoints[0]; ++i)
			{
				auto& leftOperatorY = _leftOperators[1][i + k * nSpacePoints[0]];
				auto& rightOperatorY = _rightOperators[1][i + k * nSpacePoints[0]];

				// extract inY = in[k, i, :] (this is C_{n - 1})
				for (size_t j = 0; j < nSpacePoints[1]; ++j)
					_inputCache[1][static_cast<int>(j)] = in[static_cast<int>(GetIndex(i, j, k, nSpacePoints))];

				// tmp = 0.5 * dt * L_y * C_{n - 1}
				rightOperatorY.Dot(_dotProductCache[1], _inputCache[1]);

				// add dt * (0.5 * L_x + L_y) * C_X
				for (size_t j = 0; j < nSpacePoints[1]; ++j)
				{
					const auto& Lx = spaceOperators[0][j + k * nSpacePoints[1]][i];
					if (i > 0 && i < nSpacePoints[0] - 1)
					{
						_dotProductCache[1][static_cast<int>(j)] += Real(0.5) * _inputData.deltaTime * Lx.super * in[static_cast<int>(GetIndex(i + 1, j, k, nSpacePoints))];
						_dotProductCache[1][static_cast<int>(j)] += Real(0.5) * _inputData.deltaTime * Lx.diag * in[static_cast<int>(GetIndex(i, j, k, nSpacePoints))];
						_dotProductCache[1][static_cast<int>(j)] += Real(0.5) * _inputData.deltaTime * Lx.sub * in[static_cast<int>(GetIndex(i - 1, j, k, nSpacePoints))];
					}

					const auto& Lz = spaceOperators[2][j + i * nSpacePoints[1]][k];
					if (k > 0 && k < nSpacePoints[2] - 1)
					{
						_dotProductCache[1][static_cast<int>(j)] += _inputData.deltaTime * Lz.super * in[static_cast<int>(GetIndex(i, j, k + 1, nSpacePoints))];
						_dotProductCache[1][static_cast<int>(j)] += _inputData.deltaTime * Lz.diag * in[static_cast<int>(GetIndex(i, j, k, nSpacePoints))];
						_dotProductCache[1][static_cast<int>(j)] += _inputData.deltaTime * Lz.sub * in[static_cast<int>(GetIndex(i, j, k - 1, nSpacePoints))];
					}
				}

				// add 0.5 * dt * Lx * C_X
				for (size_t j = 0; j < nSpacePoints[1]; ++j)
				{
					const auto& Lx = spaceOperators[0][j + k * nSpacePoints[1]][i];
					if (i > 0 && i < nSpacePoints[0] - 1)
					{
						_dotProductCache[1][static_cast<int>(j)] += Real(0.5) * _inputData.deltaTime * Lx.super * _outCacheX[static_cast<int>(GetIndex(i + 1, j, k, nSpacePoints))];
						_dotProductCache[1][static_cast<int>(j)] += Real(0.5) * _inputData.deltaTime * Lx.diag * _outCacheX[static_cast<int>(GetIndex(i, j, k, nSpacePoints))];
						_dotProductCache[1][static_cast<int>(j)] += Real(0.5) * _inputData.deltaTime * Lx.sub * _outCacheX[static_cast<int>(GetIndex(i - 1, j, k, nSpacePoints))];
					}
				}

				// C_Y = (I - 0.5 * dt * L_y) \ tmp
				la::TridiagonalSolver<Real> solver(leftOperatorY);
				solver.Solve(_dotProductCache[1], _solverCache[1]);

				// assign to the current output
				for (size_t j = 0; j < nSpacePoints[1]; ++j)
					_outCacheY[static_cast<int>(GetIndex(i, j, k, nSpacePoints))] = _dotProductCache[1][static_cast<int>(j)];
			}
		}

		// z-solve
		for (std::size_t i = 0; i < nSpacePoints[0]; ++i)
		{
			for (std::size_t j = 0; j < nSpacePoints[1]; ++j)
			{
				auto& leftOperatorZ = _leftOperators[2][j + i * nSpacePoints[1]];
				auto& rightOperatorZ = _rightOperators[2][j + i * nSpacePoints[1]];

				// extract inZ = in[j, i, :] (this is C_{n - 1})
				for (size_t k = 0; k < nSpacePoints[2]; ++k)
					_inputCache[2][static_cast<int>(k)] = in[static_cast<int>(GetIndex(i, j, k, nSpacePoints))];

				// tmp = 0.5 * dt * L_z * C_{n - 1}
				rightOperatorZ.Dot(_dotProductCache[2], _inputCache[2]);
				for (size_t k = 0; k < nSpacePoints[2]; ++k)
				{
					const auto& Lx = spaceOperators[0][j + k * nSpacePoints[1]][i];
					if (i > 0 && i < nSpacePoints[0] - 1)
					{
						_dotProductCache[2][static_cast<int>(k)] += Real(0.5) * _inputData.deltaTime * Lx.super * in[static_cast<int>(GetIndex(i + 1, j, k, nSpacePoints))];
						_dotProductCache[2][static_cast<int>(k)] += Real(0.5) * _inputData.deltaTime * Lx.diag * in[static_cast<int>(GetIndex(i, j, k, nSpacePoints))];
						_dotProductCache[2][static_cast<int>(k)] += Real(0.5) * _inputData.deltaTime * Lx.sub * in[static_cast<int>(GetIndex(i - 1, j, k, nSpacePoints))];
					}

					const auto& Ly = spaceOperators[1][i + k * nSpacePoints[0]][j];
					if (j > 0 && j < nSpacePoints[1] - 1)
					{
						_dotProductCache[2][static_cast<int>(k)] += Real(0.5) * _inputData.deltaTime * Ly.super * in[static_cast<int>(GetIndex(i, j + 1, k, nSpacePoints))];
						_dotProductCache[2][static_cast<int>(k)] += Real(0.5) * _inputData.deltaTime * Ly.diag * in[static_cast<int>(GetIndex(i, j, k, nSpacePoints))];
						_dotProductCache[2][static_cast<int>(k)] += Real(0.5) * _inputData.deltaTime * Ly.sub * in[static_cast<int>(GetIndex(i, j - 1, k, nSpacePoints))];
					}
				}

				// extract C_Y[k] = C_Y[j, i, :]
				// add 0.5 * dt * Lx * (C_X + C_Y)
				for (size_t k = 0; k < nSpacePoints[2]; ++k)
				{
					const auto& Lx = spaceOperators[0][j + k * nSpacePoints[1]][i];
					if (i > 0 && i < nSpacePoints[0] - 1)
					{
						_dotProductCache[2][static_cast<int>(k)] += Real(0.5) * _inputData.deltaTime * Lx.super * _outCacheX[static_cast<int>(GetIndex(i + 1, j, k, nSpacePoints))];
						_dotProductCache[2][static_cast<int>(k)] += Real(0.5) * _inputData.deltaTime * Lx.diag * _outCacheX[static_cast<int>(GetIndex(i, j, k, nSpacePoints))];
						_dotProductCache[2][static_cast<int>(k)] += Real(0.5) * _inputData.deltaTime * Lx.sub * _outCacheX[static_cast<int>(GetIndex(i - 1, j, k, nSpacePoints))];
					}

					const auto& Ly = spaceOperators[1][i + k * nSpacePoints[0]][j];
					if (j > 0 && j < nSpacePoints[1] - 1)
					{
						_dotProductCache[2][static_cast<int>(k)] += Real(0.5) * _inputData.deltaTime * Ly.super * _outCacheY[static_cast<int>(GetIndex(i, j + 1, k, nSpacePoints))];
						_dotProductCache[2][static_cast<int>(k)] += Real(0.5) * _inputData.deltaTime * Ly.diag * _outCacheY[static_cast<int>(GetIndex(i, j, k, nSpacePoints))];
						_dotProductCache[2][static_cast<int>(k)] += Real(0.5) * _inputData.deltaTime * Ly.sub * _outCacheY[static_cast<int>(GetIndex(i, j - 1, k, nSpacePoints))];
					}
				}

				// C_Y = (I - 0.5 * dt * L_y) \ tmp
				la::TridiagonalSolver<Real> solver(leftOperatorZ);
				solver.Solve(_dotProductCache[2], _solverCache[2]);

				// assign to the current output
				for (size_t k = 0; k < nSpacePoints[2]; ++k)
					out[static_cast<int>(GetIndex(i, j, k, nSpacePoints))] = _dotProductCache[2][static_cast<int>(k)];
			}
		}
	}

}	 // namespace pde

template class pde::AdiTimeDiscretizer<float>;
template class pde::AdiTimeDiscretizer<double>;
