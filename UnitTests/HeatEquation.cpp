
#include <gtest/gtest.h>

#include "Pde/Problem.h"
#include "Pde/SolverType.h"

#include <cmath>

class HeatEquationTests: public testing::Test
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
		inputData.diffusionCoefficients.fill(0.0);
		inputData.deltaTime = Real(1e-6);

		std::vector<Real> zeroVelocity(totalSize, Real(0.0));
		inputData.velocityField.fill(zeroVelocity);

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

TEST_F(HeatEquationTests, ZeroDiffusionCoefficients)
{
	for (auto solver : { pde::SolverType::ExplicitEuler, pde::SolverType::LaxWendroff })
	{
		pde::Configuration config { nSpacePoints, solver };
		// test single precision
		{
			pde::Problem<float> problem(inputDataFloat, config);
			for (size_t n = 0; n < 10; ++n)
			{
				problem.Advance();
				const auto& solution = problem.GetSolution();
				for (std::size_t k = 0; k < nSpacePoints[2]; ++k)
					for (std::size_t i = 0; i < nSpacePoints[0]; ++i)
						for (std::size_t j = 0; j < nSpacePoints[1]; ++j)
							ASSERT_FLOAT_EQ(solution[GetIndex(i, j, k)], inputDataFloat.initialCondition[GetIndex(i, j, k)]) << static_cast<int>(solver);
			}
		}

		// test double precision
		{
			pde::Problem<double> problem(inputDataDouble, config);
			for (size_t n = 0; n < 10; ++n)
			{
				problem.Advance();
				const auto& solution = problem.GetSolution();
				for (std::size_t k = 0; k < nSpacePoints[2]; ++k)
					for (std::size_t i = 0; i < nSpacePoints[0]; ++i)
						for (std::size_t j = 0; j < nSpacePoints[1]; ++j)
							ASSERT_DOUBLE_EQ(solution[GetIndex(i, j, k)], inputDataDouble.initialCondition[GetIndex(i, j, k)]) << static_cast<int>(solver);
			}
		}
	}
}

TEST_F(HeatEquationTests, ConstantSolution)
{
	for (auto solver : { pde::SolverType::ExplicitEuler, pde::SolverType::LaxWendroff })
	{
		pde::Configuration config { nSpacePoints, solver };
		// test single precision
		{
			inputDataFloat.diffusionCoefficients.fill(1.0f);
			std::fill(inputDataFloat.initialCondition.begin(), inputDataFloat.initialCondition.end(), 1.0);
			pde::Problem<float> problem(inputDataFloat, config);
			for (size_t n = 0; n < 10; ++n)
			{
				problem.Advance();
				const auto& solution = problem.GetSolution();
				for (std::size_t k = 0; k < nSpacePoints[2]; ++k)
					for (std::size_t i = 0; i < nSpacePoints[0]; ++i)
						for (std::size_t j = 0; j < nSpacePoints[1]; ++j)
							ASSERT_FLOAT_EQ(solution[GetIndex(i, j, k)], inputDataFloat.initialCondition[GetIndex(i, j, k)]) << static_cast<int>(solver);
			}
		}

		// test double precision
		{
			inputDataDouble.diffusionCoefficients.fill(1.0);
			std::fill(inputDataDouble.initialCondition.begin(), inputDataDouble.initialCondition.end(), 1.0);
			pde::Problem<double> problem(inputDataDouble, config);
			for (size_t n = 0; n < 10; ++n)
			{
				problem.Advance();
				const auto& solution = problem.GetSolution();
				for (std::size_t k = 0; k < nSpacePoints[2]; ++k)
					for (std::size_t i = 0; i < nSpacePoints[0]; ++i)
						for (std::size_t j = 0; j < nSpacePoints[1]; ++j)
							ASSERT_DOUBLE_EQ(solution[GetIndex(i, j, k)], inputDataDouble.initialCondition[GetIndex(i, j, k)]) << static_cast<int>(solver);
			}
		}
	}
}

TEST_F(HeatEquationTests, LinearSolution)
{
	for (auto solver : { pde::SolverType::ExplicitEuler, pde::SolverType::LaxWendroff })
	{
		pde::Configuration config { nSpacePoints, solver };
		// test single precision
		{
			inputDataFloat.diffusionCoefficients.fill(1.0f);
			for (std::size_t k = 0; k < nSpacePoints[2]; ++k)
				for (std::size_t i = 0; i < nSpacePoints[0]; ++i)
					for (std::size_t j = 0; j < nSpacePoints[1]; ++j)
					{
						const auto dx = static_cast<float>(i) / static_cast<float>(nSpacePoints[0] - 1);
						const auto dy = static_cast<float>(j) / static_cast<float>(nSpacePoints[1] - 1);
						const auto dz = static_cast<float>(k) / static_cast<float>(nSpacePoints[2] - 1);

						const auto x = inputDataFloat.spaceGrids[0].front() + 2.0f * inputDataFloat.spaceGrids[0].back() * dx;
						const auto y = inputDataFloat.spaceGrids[1].front() + 2.0f * inputDataFloat.spaceGrids[1].back() * dy;
						const auto z = inputDataFloat.spaceGrids[2].front() + 2.0f * inputDataFloat.spaceGrids[2].back() * dz;
						inputDataFloat.initialCondition[GetIndex(i, j, k)] = x + y + z;
					}
			pde::Problem<float> problem(inputDataFloat, config);
			for (size_t n = 0; n < 10; ++n)
			{
				problem.Advance();
				const auto& solution = problem.GetSolution();
				for (std::size_t k = 1; k < nSpacePoints[2] - 1; ++k)
					for (std::size_t i = 1; i < nSpacePoints[0] - 1; ++i)
						for (std::size_t j = 1; j < nSpacePoints[1] - 1; ++j)
							ASSERT_NEAR(solution[GetIndex(i, j, k)], inputDataFloat.initialCondition[GetIndex(i, j, k)], 2.5e-4);
			}
		}

		// test double precision
		{
			inputDataDouble.diffusionCoefficients.fill(1.0);
			for (std::size_t k = 0; k < nSpacePoints[2]; ++k)
				for (std::size_t i = 0; i < nSpacePoints[0]; ++i)
					for (std::size_t j = 0; j < nSpacePoints[1]; ++j)
					{
						const auto dx = static_cast<double>(i) / static_cast<double>(nSpacePoints[0] - 1);
						const auto dy = static_cast<double>(j) / static_cast<double>(nSpacePoints[1] - 1);
						const auto dz = static_cast<double>(k) / static_cast<double>(nSpacePoints[2] - 1);

						const auto x = inputDataDouble.spaceGrids[0].front() + (inputDataDouble.spaceGrids[0].back() - inputDataDouble.spaceGrids[0].front()) * dx;
						const auto y = inputDataDouble.spaceGrids[1].front() + (inputDataDouble.spaceGrids[1].back() - inputDataDouble.spaceGrids[1].front()) * dy;
						const auto z = inputDataDouble.spaceGrids[2].front() + (inputDataDouble.spaceGrids[2].back() - inputDataDouble.spaceGrids[2].front()) * dz;
						inputDataDouble.initialCondition[GetIndex(i, j, k)] = x + y + z;
					}
			pde::Problem<double> problem(inputDataDouble, config);
			for (size_t n = 0; n < 10; ++n)
			{
				problem.Advance();
				const auto& solution = problem.GetSolution();
				for (std::size_t k = 1; k < nSpacePoints[2] - 1; ++k)
					for (std::size_t i = 1; i < nSpacePoints[0] - 1; ++i)
						for (std::size_t j = 1; j < nSpacePoints[1] - 1; ++j)
							EXPECT_NEAR(solution[GetIndex(i, j, k)], inputDataDouble.initialCondition[GetIndex(i, j, k)], 2.5e-4) << n << " | " << i << " | " << j << " | " << k;
			}
		}
	}
}
