
#include <gtest/gtest.h>

#include "Pde/Problem.h"
#include "Pde/SolverType.h"

#include <cmath>

class AdvectionDiffusionTests: public testing::Test
{
public:
	[[nodiscard]] constexpr std::size_t GetIndex(const std::size_t i, const std::size_t j, const std::size_t k) const noexcept
	{
		constexpr pde::Configuration config { nSpacePoints };
		return config.GetIndex(i, j, k);
	}

	template<typename Real>
	[[nodiscard]] constexpr Real InitialCondition(const Real x, const Real y, const Real z) const noexcept
	{
		return std::exp(-Real(2.0) * (x * x + y * y + z * z));
		//		return std::sin(x + y + 0 * z);
	}

	void SetUp() override
	{
		SetUpInputData(inputDataFloat);
		SetUpInputData(inputDataDouble);
	}

	template<typename Real>
	void SetUpInputData(pde::InputData<Real>& inputData)
	{
		inputData.diffusionCoefficients.fill(Real(0.1));
		inputData.deltaTime = Real(1e-5);

		std::vector<Real> velocity(totalSize, Real(0.01));
		inputData.velocityField.fill(velocity);

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

	static constexpr std::array<size_t, 3> nSpacePoints = { 16, 16, 16 };
	static constexpr size_t totalSize = nSpacePoints[0] * nSpacePoints[1] * nSpacePoints[2];

	pde::InputData<float> inputDataFloat {};
	pde::InputData<double> inputDataDouble {};
};

TEST_F(AdvectionDiffusionTests, Basic)
{
	for (auto solver : { pde::SolverType::ExplicitEuler, pde::SolverType::LaxWendroff })
	{
		pde::Configuration config { nSpacePoints, solver };
		// test single precision
		{
			pde::Problem<float> problem(inputDataFloat, config);

			auto previousMax = std::numeric_limits<float>::quiet_NaN();
			for (size_t n = 0; n < 10; ++n)
			{
				problem.Advance();
				const auto& solution = problem.GetSolution();

				// check there's no instability
				auto currentMax = *std::max_element(solution.begin(), solution.end());
				if (!std::isfinite(previousMax))
					previousMax = currentMax;
				else
				{
					ASSERT_LT(currentMax, previousMax * 1.0002f);
				}
			}
		}

		// test double precision
		{
			pde::Problem<double> problem(inputDataDouble, config);
			auto previousMax = std::numeric_limits<double>::quiet_NaN();
			for (size_t n = 0; n < 10; ++n)
			{
				problem.Advance();
				const auto& solution = problem.GetSolution();

				// check there's no instability
				auto currentMax = *std::max_element(solution.begin(), solution.end());
				if (!std::isfinite(previousMax))
					previousMax = currentMax;
				else
				{
					ASSERT_LT(currentMax, previousMax * 1.0002);
				}
			}
		}
	}
}

TEST_F(AdvectionDiffusionTests, ConsistencyWithLinearOperator)
{
	for (auto solver : { pde::SolverType::ExplicitEuler, pde::SolverType::LaxWendroff })
	{
		pde::Configuration config { nSpacePoints, solver };
		// test single precision
		{
			pde::Problem<float> problem(inputDataFloat, config);
			pde::LinearOperatorProblem<float> linearOperatorProblem(inputDataFloat, config);

			for (size_t n = 0; n < 10; ++n)
			{
				problem.Advance();
				const auto& solution = problem.GetSolution();
				linearOperatorProblem.Advance();
				const auto& linearOperatorSolution = linearOperatorProblem.GetSolution();

				for (std::size_t k = 1; k < nSpacePoints[2] - 1; ++k)
				{
					for (std::size_t i = 1; i < nSpacePoints[0] - 1; ++i)
					{
						for (std::size_t j = 1; j < nSpacePoints[1] - 1; ++j)
						{
							EXPECT_NEAR(solution[GetIndex(i, j, k)], linearOperatorSolution[GetIndex(i, j, k)], 1e-6) << i << "|" << j << "|" << k << " | --- " << n;
						}
					}
				}
			}
		}

		// test double precision
		{
			pde::Problem<double> problem(inputDataDouble, config);
			pde::LinearOperatorProblem<double> linearOperatorProblem(inputDataDouble, config);

			for (size_t n = 0; n < 1; ++n)
			{
				problem.Advance();
				const auto& solution = problem.GetSolution();
				linearOperatorProblem.Advance();
				const auto& linearOperatorSolution = linearOperatorProblem.GetSolution();

				for (std::size_t k = 1; k < nSpacePoints[2] - 1; ++k)
				{
					for (std::size_t i = 1; i < nSpacePoints[0] - 1; ++i)
					{
						for (std::size_t j = 1; j < nSpacePoints[1] - 1; ++j)
						{
							EXPECT_NEAR(solution[GetIndex(i, j, k)], linearOperatorSolution[GetIndex(i, j, k)], 5e-9) << i << "|" << j << "|" << k << " | --- " << n;
						}
					}
				}
			}
		}
	}
}
