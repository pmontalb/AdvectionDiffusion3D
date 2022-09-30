
#include "NpyCpp/Npy++/Npy++.h"
#include "Problem.h"

#include <cmath>
#include <fstream>
#include <iostream>

static constexpr std::array<size_t, 3> nSpacePoints = { 16, 16, 16 };
static constexpr size_t totalSize = nSpacePoints[0] * nSpacePoints[1] * nSpacePoints[2];

[[nodiscard]] static constexpr std::size_t GetIndex(const std::size_t i, const std::size_t j, const std::size_t k) noexcept { return pde::GetIndex(i, j, k, nSpacePoints); }
template<typename Real>
[[nodiscard]] static constexpr Real InitialCondition(const Real x, const Real y, const Real z) noexcept
{
	return std::exp(-Real(2.0) * (x * x + y * y + z * z));
	//		return std::sin(x + y + 0 * z);
}

template<typename Real>
static void SetUpInputData(pde::InputData<Real>& inputData)
{
	inputData.diffusionCoefficients.fill(Real(0.5));
	inputData.deltaTime = Real(1e-2);

	inputData.velocityField[0] = std::vector<Real>(totalSize, Real(1));
	inputData.velocityField[1] = std::vector<Real>(totalSize, Real(1));
	inputData.velocityField[2] = std::vector<Real>(totalSize, Real(1));

	for (size_t n = 0; n < inputData.spaceGrids.size(); ++n)
	{
		inputData.spaceGrids[n].resize(nSpacePoints[n]);
		for (size_t i = 0; i < inputData.spaceGrids[n].size(); ++i)
			inputData.spaceGrids[n][i] = -Real(1.0) + Real(2.0) * static_cast<Real>(i) / static_cast<Real>(nSpacePoints[n] - 1);
	}
	inputData.initialCondition.resize(totalSize);
	for (std::size_t k = 0; k < nSpacePoints[2]; ++k)
		for (std::size_t i = 0; i < nSpacePoints[0]; ++i)
			for (std::size_t j = 0; j < nSpacePoints[1]; ++j)
				inputData.initialCondition[GetIndex(i, j, k)] = InitialCondition(inputData.spaceGrids[0][i], inputData.spaceGrids[1][j], inputData.spaceGrids[2][k]);

	for (std::size_t k = 0; k < nSpacePoints[2]; ++k)
	{
		for (std::size_t j = 0; j < nSpacePoints[1]; ++j)
		{
			inputData.initialCondition[GetIndex(0, j, k)] = inputData.initialCondition[GetIndex(1, j, k)];
			inputData.initialCondition[GetIndex(nSpacePoints[0] - 1, j, k)] = inputData.initialCondition[GetIndex(nSpacePoints[0] - 2, j, k)];
		}
	}
	for (std::size_t k = 0; k < nSpacePoints[2]; ++k)
	{
		for (std::size_t i = 0; i < nSpacePoints[1]; ++i)
		{
			inputData.initialCondition[GetIndex(i, 0, k)] = inputData.initialCondition[GetIndex(i, 1, k)];
			inputData.initialCondition[GetIndex(i, nSpacePoints[1] - 1, k)] = inputData.initialCondition[GetIndex(i, nSpacePoints[1] - 2, k)];
		}
	}
	for (std::size_t i = 0; i < nSpacePoints[0]; ++i)
	{
		for (std::size_t j = 0; j < nSpacePoints[1]; ++j)
		{
			inputData.initialCondition[GetIndex(i, j, 0)] = inputData.initialCondition[GetIndex(i, j, 1)];
			inputData.initialCondition[GetIndex(i, j, nSpacePoints[2] - 1)] = inputData.initialCondition[GetIndex(i, j, nSpacePoints[2] - 2)];
		}
	}
}

template<typename Real>
void Print(const std::vector<Real>& solution, const std::string& label = "solution")
{
	std::cerr << label << "= np.array([";
	for (std::size_t k = 0; k < nSpacePoints[2]; ++k)
		for (std::size_t i = 0; i < nSpacePoints[0]; ++i)
			for (std::size_t j = 0; j < nSpacePoints[1]; ++j)
				std::cerr << solution[GetIndex(i, j, k)] << ", ";
	std::cerr << "])" << std::endl;
}

template<typename VectorT>
void ToFile(std::ofstream& ofs, const VectorT& solution, const std::string& label = "solution")
{
	ofs << label << "= np.array([";
	for (std::size_t k = 0; k < nSpacePoints[2]; ++k)
		for (std::size_t i = 0; i < nSpacePoints[0]; ++i)
			for (std::size_t j = 0; j < nSpacePoints[1]; ++j)
				ofs << solution[GetIndex(i, j, k)] << ", ";
	ofs << "])" << std::endl;
}

inline void StabilityAnalysis()
{
	using Real = double;
	pde::InputData<Real> inputData;
	SetUpInputData(inputData);
	inputData.deltaTime *= 10.0;

	const std::string fileNameEE = "/home/raiden/programming/AdvectionDiffusion3D/cmake-build-gcc-debug/explicitEuler.txt";
	const std::string fileNameIE = "/home/raiden/programming/AdvectionDiffusion3D/cmake-build-gcc-debug/implicitEuler.txt";
	const std::string fileNameCN = "/home/raiden/programming/AdvectionDiffusion3D/cmake-build-gcc-debug/crankNicolson.txt";
	const std::string fileNameLW = "/home/raiden/programming/AdvectionDiffusion3D/cmake-build-gcc-debug/laxWendroff.txt";
	const std::string fileNameAD = "/home/raiden/programming/AdvectionDiffusion3D/cmake-build-gcc-debug/adi.txt";

	pde::LinearOperatorProblem<Real> problemEE(inputData);
	pde::LinearOperatorProblem<Real> problemIE(inputData);
	pde::LinearOperatorProblem<Real> problemCN(inputData);
	pde::LinearOperatorProblem<Real> problemLW(inputData);
	pde::LinearOperatorProblem<Real> problemAD(inputData);

	const auto worker = [&](pde::LinearOperatorProblem<Real>& problem, const pde::SolverType solverType, const std::string& fileName, const bool append)
	{
		problem.Advance(solverType);
		const auto& solution = problem.GetSolution();
		std::vector<double> solutionCopy { solution.begin(), solution.end() };
		if (!append)
			npypp::Save(fileName, solutionCopy, { inputData.spaceGrids[0].size(), inputData.spaceGrids[1].size(), inputData.spaceGrids[2].size() }, "w");
		else
			npypp::Save(fileName, solutionCopy, { inputData.spaceGrids[0].size(), inputData.spaceGrids[1].size(), inputData.spaceGrids[2].size() }, "a");
	};

	for (size_t n = 0; n < 600; ++n)
	{
		worker(problemEE, pde::SolverType::ExplicitEuler, fileNameEE, n > 0);
		worker(problemIE, pde::SolverType::ImplicitEuler, fileNameIE, n > 0);
		worker(problemCN, pde::SolverType::CrankNicolson, fileNameCN, n > 0);
		worker(problemLW, pde::SolverType::LaxWendroff, fileNameLW, n > 0);
		worker(problemAD, pde::SolverType::ADI, fileNameAD, n > 0);
	}
}

inline void SingleRun()
{
	using Real = double;
	pde::InputData<Real> inputData;
	SetUpInputData(inputData);

	const std::string fileName = "/home/raiden/programming/AdvectionDiffusion3D/cmake-build-gcc-debug/sol.txt";
	pde::LinearOperatorProblem<Real> problem(inputData);
	for (size_t n = 0; n < 600; ++n)
	{
		problem.Advance(pde::SolverType::ADI);
		const auto& solution = problem.GetSolution();
		std::vector<double> solutionCopy { solution.begin(), solution.end() };
		if (n == 0)
			npypp::Save(fileName, solutionCopy, { inputData.spaceGrids[0].size(), inputData.spaceGrids[1].size(), inputData.spaceGrids[2].size() }, "w");
		else
			npypp::Save(fileName, solutionCopy, { inputData.spaceGrids[0].size(), inputData.spaceGrids[1].size(), inputData.spaceGrids[2].size() }, "a");
	}
}

int main(int /*argc*/, char** /*argv*/) { StabilityAnalysis(); }
