
#include "Problem.h"

#include "Profiler.h"

#include <cmath>
#include <memory>

[[nodiscard]] inline constexpr std::size_t GetIndex(const std::size_t i, const std::size_t j, const std::size_t k, const std::size_t Nx, const std::size_t Ny, const std::size_t Nz) noexcept
{
	std::array<size_t, 3> nSpacePoints = { { Nx, Ny, Nz } };
	return pde::GetIndex(i, j, k, nSpacePoints);
}

template<typename Real>
[[nodiscard]] inline constexpr Real InitialCondition(const Real x, const Real y, const Real z) noexcept
{
	return std::exp(-Real(2.0) * (x * x + y * y + z * z));
}

template<typename Real>
inline void SetUpInputData(pde::InputData<Real>& inputData, const size_t Nx, const size_t Ny, const size_t Nz)
{
	inputData.diffusionCoefficients.fill(Real(0.));
	inputData.deltaTime = Real(1e-2);

	const auto totalSize = Nx * Ny * Nz;
	inputData.velocityField[0] = std::vector<Real>(totalSize, Real(1));
	inputData.velocityField[1] = std::vector<Real>(totalSize, Real(1));
	inputData.velocityField[2] = std::vector<Real>(totalSize, Real(1));

	inputData.spaceGrids[0].resize(Nx);
	inputData.spaceGrids[1].resize(Ny);
	inputData.spaceGrids[2].resize(Nz);
	for (size_t i = 0; i < inputData.spaceGrids[0].size(); ++i)
		inputData.spaceGrids[0][i] = -Real(1.0) + Real(2.0) * static_cast<Real>(i) / static_cast<Real>(Nx - 1);
	for (size_t i = 0; i < inputData.spaceGrids[1].size(); ++i)
		inputData.spaceGrids[1][i] = -Real(1.0) + Real(2.0) * static_cast<Real>(i) / static_cast<Real>(Ny - 1);
	for (size_t i = 0; i < inputData.spaceGrids[2].size(); ++i)
		inputData.spaceGrids[2][i] = -Real(1.0) + Real(2.0) * static_cast<Real>(i) / static_cast<Real>(Nz - 1);

	inputData.initialCondition.resize(totalSize);
	for (std::size_t k = 0; k < Nz; ++k)
		for (std::size_t i = 0; i < Ny; ++i)
			for (std::size_t j = 0; j < Nx; ++j)
				inputData.initialCondition[GetIndex(i, j, k, Nx, Ny, Nz)] = InitialCondition(inputData.spaceGrids[0][i], inputData.spaceGrids[1][j], inputData.spaceGrids[2][k]);
}

struct Config: public perf::Config
{
	std::size_t Nx = 0;
	std::size_t Ny = 0;
	std::size_t Nz = 0;
	pde::SolverType solverType = pde::SolverType::Null;
};

template<typename Real, typename ProblemT>
class PdeProfiler: public perf::Profiler<PdeProfiler<Real, ProblemT>, Config>
{
	friend class perf::Profiler<PdeProfiler<Real, ProblemT>, Config>;

public:
	using perf::Profiler<PdeProfiler<Real, ProblemT>, Config>::Profiler;

private:
	inline void OnStartImpl() noexcept
	{
		SetUpInputData(_inputData, this->_config.Nx, this->_config.Ny, this->_config.Nz);
		_solver = std::make_unique<ProblemT>(_inputData);

		const auto radius = Real(0.8);
		const auto sourceStrength = Real(0.1);
		_sourceTerm.resize(static_cast<long>(_inputData.initialCondition.size()));
		_sourceTerm.fill(0.0);
		for (std::size_t k = 0; k < this->_config.Nz; ++k)
			for (std::size_t i = 0; i < this->_config.Nx; ++i)
				for (std::size_t j = 0; j < this->_config.Ny; ++j)
				{
					const auto x = _inputData.spaceGrids[0][i] - _inputData.spaceGrids[0][this->_config.Nx / 2];
					const auto y = _inputData.spaceGrids[1][j] - _inputData.spaceGrids[1][this->_config.Ny / 2];
					const auto z = 0;
					if (x * x + y * y + z * z > radius * radius)
						continue;
					_sourceTerm[static_cast<int>(GetIndex(i, j, k, this->_config.Nx, this->_config.Ny, this->_config.Nz))] = sourceStrength;
				}
	}

	inline void OnEndImpl() noexcept { std::cout << " done!" << std::endl; }

	inline void RunImpl() noexcept
	{
		_solver->Advance(this->_config.solverType);
//		_solver->Advance(this->_config.solverType, this->_sourceTerm);

	}

private:
	pde::InputData<Real> _inputData;
	std::unique_ptr<ProblemT> _solver = nullptr;
	Eigen::VectorX<Real> _sourceTerm;
};

int main(int /*argc*/, char** /*argv*/)
{
		constexpr std::size_t nSpacePoints[] = { 90, 100, 120, 150 };
		constexpr pde::SolverType inlineSolvers[] = { pde::SolverType::ExplicitEuler, pde::SolverType::LaxWendroff };
		constexpr pde::SolverType operatorSolvers[] = { pde::SolverType::ExplicitEuler, pde::SolverType::LaxWendroff, pde::SolverType::ADI };
//		constexpr pde::SolverType operatorSolvers[] = { pde::SolverType::ImplicitEuler, pde::SolverType::CrankNicolson };

	Config config;
	config.nIterations = 100;
	config.nIterationsPerCycle = 1;
	config.nWarmUpIterations = 1;

	for (auto N : nSpacePoints)
	{
		config.Nx = N;
		config.Ny = N;
		config.Nz = N;
		for (auto solverType : inlineSolvers)
		{
			config.solverType = solverType;
			{
				PdeProfiler<float, pde::Problem<float>> profiler(config);
				profiler.Profile();

				std::cout << "N=" << N << " | Precision: Single | Discretization: Inline | Solver:" << pde::ToString(solverType) << std::endl;
				profiler.GetPerformance().Print();
			}

			{
				PdeProfiler<double, pde::Problem<double>> profiler(config);
				profiler.Profile();

				std::cout << "N=" << N << " | Precision: Double | Discretization: Inline | Solver:" << pde::ToString(solverType) << std::endl;
				profiler.GetPerformance().Print();
			}
		}

		for (auto solverType : operatorSolvers)
		{
			config.solverType = solverType;
			{
				PdeProfiler<float, pde::LinearOperatorProblem<float>> profiler(config);
				profiler.Profile();

				std::cout << "N=" << N << " | Precision: Single | Discretization: Operator | Solver:" << pde::ToString(solverType) << std::endl;
				profiler.GetPerformance().Print();
			}

			{
				PdeProfiler<double, pde::LinearOperatorProblem<double>> profiler(config);
				profiler.Profile();

				std::cout << "N=" << N << " | Precision: Double | Discretization: Operator | Solver:" << pde::ToString(solverType) << std::endl;
				profiler.GetPerformance().Print();
			}
		}
	}
}
