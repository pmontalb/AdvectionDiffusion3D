
#include "AdiTimeDiscretizer.h"


namespace pde
{
	template<typename Real>
	AdiTimeDiscretizer<Real>::AdiTimeDiscretizer(const InputData<Real>& inputData, const AdiSpaceDiscretizer<Real>& spaceDiscretizer) : _inputData(inputData), _spaceDiscretizer(spaceDiscretizer)
	{
		const auto& spaceDiscretizations = _spaceDiscretizer.GetSpaceDiscretizations();

		for (size_t i = 0; i < _leftOperators.size(); ++i)
			_leftOperators[i].resize(spaceDiscretizations[i].size());
		for (size_t i = 0; i < _leftOperators.size(); ++i)
		{
			for (size_t j = 0; j < _leftOperators[i].size(); ++j)
			{
				auto& matrix = _leftOperators[i][j];
				matrix.Resize(spaceDiscretizations[i][0].Size());
			}
		}

		for (size_t i = 0; i < _rightOperators.size(); ++i)
			_rightOperators[i].resize(spaceDiscretizations[i].size());
		for (size_t i = 0; i < _rightOperators.size(); ++i)
		{
			for (size_t j = 0; j < _rightOperators[i].size(); ++j)
			{
				auto& matrix = _rightOperators[i][j];
				matrix.Resize(spaceDiscretizations[i][0].Size());
			}
		}

		for (size_t i = 0; i < _solverCache.size(); ++i)
		{
			_solverCache[i].resize(static_cast<long>(_spaceDiscretizer.GetNumberOfSpacePoints()[i]));
			_dotProductCache[i].resize(static_cast<long>(_spaceDiscretizer.GetNumberOfSpacePoints()[i]));
			_inputCache[i].resize(static_cast<long>(_spaceDiscretizer.GetNumberOfSpacePoints()[i]));
			_intermediateCache[i].resize(static_cast<long>(_spaceDiscretizer.GetNumberOfSpacePoints()[i]));
		}
	}

	template<typename Real>
	bool AdiTimeDiscretizer<Real>::Precompute() noexcept
	{
		const auto& nSpacePoints = _spaceDiscretizer.GetNumberOfSpacePoints();
		const auto& spaceOpeartors = _spaceDiscretizer.GetSpaceDiscretizations();

		static constexpr auto one = Real(1.0);
		static constexpr auto half = Real(0.5);

		for (std::size_t k = 0; k < nSpacePoints[2]; ++k)
		{
			for (std::size_t i = 0; i < nSpacePoints[0]; ++i)
			{
				for (std::size_t j = 0; j < nSpacePoints[1]; ++j)
				{
					auto& spaceOperatorX = spaceOpeartors[0][j + k * nSpacePoints[1]][i];
					auto& spaceOperatorY = spaceOpeartors[1][i + k * nSpacePoints[0]][j];
					auto& spaceOperatorZ = spaceOpeartors[2][j + i * nSpacePoints[1]][k];

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

					// A_x = I + 0.5 * dt * (L_y + Lz)
					rightOperatorX = {
						half * _inputData.deltaTime * (spaceOperatorY.sub + spaceOperatorZ.sub),
						one + half * _inputData.deltaTime * (spaceOperatorY.diag + spaceOperatorZ.diag),
						half * _inputData.deltaTime * (spaceOperatorY.super + spaceOperatorZ.super),
					};

					// A_y = 0.5 * dt * L_y
					rightOperatorY = {
						half * _inputData.deltaTime * spaceOperatorY.sub,
						half * _inputData.deltaTime * spaceOperatorY.diag,
						half * _inputData.deltaTime * spaceOperatorY.super,
					};

					// A_z = 0.5 * dt * L_z
					rightOperatorZ = {
						half * _inputData.deltaTime * spaceOperatorZ.sub,
						half * _inputData.deltaTime * spaceOperatorZ.diag,
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

				// tmp = (I + 0.5 * dt * (L_y + Lz)) * C_{n - 1}
				rightOperatorX.Dot(_dotProductCache[0], _inputCache[0]);
				// C_X = (I - 0.5 * dt * L_x) \ tmp
				la::TridiagonalSolver<Real> solver(leftOperatorX);
				solver.Solve(_dotProductCache[0], _solverCache[0]);

				// assign to the current output
				for (size_t i = 0; i < nSpacePoints[0]; ++i)
					out[static_cast<int>(GetIndex(i, j, k, nSpacePoints))] = _dotProductCache[0][static_cast<int>(i)];
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

				// extract C_X[j] = C_X[k, i, :]
				for (size_t j = 0; j < nSpacePoints[1]; ++j)
					_intermediateCache[1][static_cast<int>(j)] = out[static_cast<int>(GetIndex(i, j, k, nSpacePoints))];

				// add C_X
				_dotProductCache[1].noalias() += _intermediateCache[1];
				// C_Y = (I - 0.5 * dt * L_y) \ tmp
				la::TridiagonalSolver<Real> solver(leftOperatorY);
				solver.Solve(_dotProductCache[1], _solverCache[1]);

				// assign to the current output
				for (size_t j = 0; j < nSpacePoints[1]; ++j)
					out[static_cast<int>(GetIndex(i, j, k, nSpacePoints))] = _dotProductCache[1][static_cast<int>(j)];
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

				// extract C_Y[k] = C_Y[j, i, :]
				for (size_t k = 0; k < nSpacePoints[2]; ++k)
					_intermediateCache[2][static_cast<int>(k)] = out[static_cast<int>(GetIndex(i, j, k, nSpacePoints))];

				// add C_Y
				_dotProductCache[2].noalias() += _intermediateCache[2];

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
