
#include <gtest/gtest.h>

#include "Pde/Problem.h"
#include "Pde/SolverType.h"

#include <cmath>

class TransportEquationTests: public testing::Test
{
public:
	[[nodiscard]] constexpr std::size_t GetIndex(const std::size_t i, const std::size_t j, const std::size_t k) const noexcept
	{
		return pde::GetIndex(i, j, k, nSpacePoints);
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
		inputData.deltaTime = Real(1.0);

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

TEST_F(TransportEquationTests, ZeroTransportCoefficients)
{
	for (auto solver : { pde::SolverType::ExplicitEuler, pde::SolverType::LaxWendroff })
	{
		// test single precision
		{
			pde::Problem<float> problem(inputDataFloat);
			for (size_t n = 0; n < 10; ++n)
			{
				problem.Advance(solver);
				const auto& solution = problem.GetSolution();
				for (std::size_t k = 0; k < nSpacePoints[2]; ++k)
					for (std::size_t i = 0; i < nSpacePoints[0]; ++i)
						for (std::size_t j = 0; j < nSpacePoints[1]; ++j)
							ASSERT_FLOAT_EQ(solution[GetIndex(i, j, k)], inputDataFloat.initialCondition[GetIndex(i, j, k)]) << static_cast<int>(solver);
			}
		}

		// test double precision
		{
			pde::Problem<double> problem(inputDataDouble);
			for (size_t n = 0; n < 10; ++n)
			{
				problem.Advance(solver);
				const auto& solution = problem.GetSolution();
				for (std::size_t k = 0; k < nSpacePoints[2]; ++k)
					for (std::size_t i = 0; i < nSpacePoints[0]; ++i)
						for (std::size_t j = 0; j < nSpacePoints[1]; ++j)
							ASSERT_DOUBLE_EQ(solution[GetIndex(i, j, k)], inputDataDouble.initialCondition[GetIndex(i, j, k)]) << static_cast<int>(solver);
			}
		}
	}
}

TEST_F(TransportEquationTests, ConstantSolution)
{
	for (auto solver : { pde::SolverType::ExplicitEuler, pde::SolverType::LaxWendroff })
	{
		// test single precision
		{
			std::vector<float> nonZeroVelocity(totalSize, 1.0f);
			inputDataFloat.velocityField.fill(nonZeroVelocity);
			std::fill(inputDataFloat.initialCondition.begin(), inputDataFloat.initialCondition.end(), 1.0);
			pde::Problem<float> problem(inputDataFloat);
			for (size_t n = 0; n < 10; ++n)
			{
				problem.Advance(solver);
				const auto& solution = problem.GetSolution();
				for (std::size_t k = 0; k < nSpacePoints[2]; ++k)
					for (std::size_t i = 0; i < nSpacePoints[0]; ++i)
						for (std::size_t j = 0; j < nSpacePoints[1]; ++j)
							ASSERT_FLOAT_EQ(solution[GetIndex(i, j, k)], inputDataFloat.initialCondition[GetIndex(i, j, k)]) << static_cast<int>(solver);
			}
		}

		// test double precision
		{
			std::vector<double> nonZeroVelocity(totalSize, 1.0);
			inputDataDouble.velocityField.fill(nonZeroVelocity);
			std::fill(inputDataDouble.initialCondition.begin(), inputDataDouble.initialCondition.end(), 1.0);
			pde::Problem<double> problem(inputDataDouble);
			for (size_t n = 0; n < 10; ++n)
			{
				problem.Advance(solver);
				const auto& solution = problem.GetSolution();
				for (std::size_t k = 0; k < nSpacePoints[2]; ++k)
					for (std::size_t i = 0; i < nSpacePoints[0]; ++i)
						for (std::size_t j = 0; j < nSpacePoints[1]; ++j)
							ASSERT_DOUBLE_EQ(solution[GetIndex(i, j, k)], inputDataDouble.initialCondition[GetIndex(i, j, k)]) << static_cast<int>(solver);
			}
		}
	}
}

TEST_F(TransportEquationTests, TransportAlongAxes)
{
	for (auto solver : { pde::SolverType::ExplicitEuler, pde::SolverType::LaxWendroff })
	{
		// test single precision
		{
			for (size_t axisIdx = 0; axisIdx < 3; ++axisIdx)
			{
				// NB: the error is proportional to v * dt
				inputDataFloat.deltaTime = 1e-2f;
				const auto velocity = 0.1f;
				inputDataFloat.velocityField.fill(std::vector<float>(totalSize, 0.0));
				inputDataFloat.velocityField[axisIdx] = std::vector<float>(totalSize, velocity);

				pde::Problem<float> problem(inputDataFloat);
				for (size_t n = 0; n < 20; ++n)
				{
					problem.Advance(solver);
					const auto& solution = problem.GetSolution();
					for (std::size_t k = 1; k < nSpacePoints[2] - 1; ++k)
					{
						for (std::size_t i = 1; i < nSpacePoints[0] - 1; ++i)
						{
							for (std::size_t j = 1; j < nSpacePoints[1] - 1; ++j)
							{
								ASSERT_NE(solution[GetIndex(i, j, k)], inputDataFloat.initialCondition[GetIndex(i, j, k)]);

								if (axisIdx == 0)
								{
									const auto characteristicX = inputDataFloat.spaceGrids[0][i] + velocity * inputDataFloat.deltaTime * static_cast<float>(n + 1);
									const auto expected = InitialCondition(characteristicX, inputDataFloat.spaceGrids[1][j], inputDataFloat.spaceGrids[2][k]);
									const auto actual = solution[GetIndex(i, j, k)];
									EXPECT_NEAR(actual, expected, 6.6e-3) << n << " | " << i << " | " << j << " | " << k;
								}
								else if (axisIdx == 1)
								{
									const auto characteristicY = inputDataFloat.spaceGrids[1][j] + velocity * inputDataFloat.deltaTime * static_cast<float>(n + 1);
									const auto expected = InitialCondition(inputDataFloat.spaceGrids[0][i], characteristicY, inputDataFloat.spaceGrids[2][k]);
									const auto actual = solution[GetIndex(i, j, k)];
									ASSERT_NEAR(actual, expected, 6.6e-3) << n << " | " << i << " | " << j << " | " << k;
								}
								else if (axisIdx == 2)
								{
									const auto characteristicZ = inputDataFloat.spaceGrids[2][k] + velocity * inputDataFloat.deltaTime * static_cast<float>(n + 1);
									const auto expected = InitialCondition(inputDataFloat.spaceGrids[0][i], inputDataFloat.spaceGrids[1][j], characteristicZ);
									const auto actual = solution[GetIndex(i, j, k)];
									ASSERT_NEAR(actual, expected, 6.6e-3) << n << " | " << i << " | " << j << " | " << k;
								}
							}
						}
					}
				}
			}
		}

		// test double precision
		{
			for (size_t axisIdx = 0; axisIdx < 3; ++axisIdx)
			{
				// NB: the error is proportional to v * dt
				inputDataDouble.deltaTime = 1e-2;
				const auto velocity = 0.1;
				inputDataDouble.velocityField.fill(std::vector<double>(totalSize, 0.0));
				inputDataDouble.velocityField[axisIdx] = std::vector<double>(totalSize, velocity);

				pde::Problem<double> problem(inputDataDouble);
				for (size_t n = 0; n < 20; ++n)
				{
					problem.Advance(solver);
					const auto& solution = problem.GetSolution();
					for (std::size_t k = 1; k < nSpacePoints[2] - 1; ++k)
					{
						for (std::size_t i = 1; i < nSpacePoints[0] - 1; ++i)
						{
							for (std::size_t j = 1; j < nSpacePoints[1] - 1; ++j)
							{
								ASSERT_NE(solution[GetIndex(i, j, k)], inputDataDouble.initialCondition[GetIndex(i, j, k)]);

								if (axisIdx == 0)
								{
									const auto characteristicX = inputDataDouble.spaceGrids[0][i] + velocity * inputDataDouble.deltaTime * static_cast<double>(n + 1);
									const auto expected = InitialCondition(characteristicX, inputDataDouble.spaceGrids[1][j], inputDataDouble.spaceGrids[2][k]);
									const auto actual = solution[GetIndex(i, j, k)];
									EXPECT_NEAR(actual, expected, 6.6e-3) << n << " | " << i << " | " << j << " | " << k;
								}
								else if (axisIdx == 1)
								{
									const auto characteristicY = inputDataDouble.spaceGrids[1][j] + velocity * inputDataDouble.deltaTime * static_cast<double>(n + 1);
									const auto expected = InitialCondition(inputDataDouble.spaceGrids[0][i], characteristicY, inputDataDouble.spaceGrids[2][k]);
									const auto actual = solution[GetIndex(i, j, k)];
									ASSERT_NEAR(actual, expected, 6.6e-3) << n << " | " << i << " | " << j << " | " << k;
								}
								else if (axisIdx == 2)
								{
									const auto characteristicZ = inputDataDouble.spaceGrids[2][k] + velocity * inputDataDouble.deltaTime * static_cast<double>(n + 1);
									const auto expected = InitialCondition(inputDataDouble.spaceGrids[0][i], inputDataDouble.spaceGrids[1][j], characteristicZ);
									const auto actual = solution[GetIndex(i, j, k)];
									ASSERT_NEAR(actual, expected, 6.6e-3) << n << " | " << i << " | " << j << " | " << k;
								}
							}
						}
					}
				}
			}
		}
	}
}

class TransportEquationTestsWithLinearOpeator: public TransportEquationTests
{
};

TEST_F(TransportEquationTestsWithLinearOpeator, ZeroTransportCoefficients)
{
	for (auto solver : { pde::SolverType::ExplicitEuler, pde::SolverType::LaxWendroff, pde::SolverType::ImplicitEuler, pde::SolverType::CrankNicolson  })
	{
		// test single precision
		{
			pde::LinearOperatorProblem<float> problem(inputDataFloat);
			for (size_t n = 0; n < 10; ++n)
			{
				problem.Advance(solver);
				const auto& solution = problem.GetSolution();
				for (std::size_t k = 0; k < nSpacePoints[2]; ++k)
					for (std::size_t i = 0; i < nSpacePoints[0]; ++i)
						for (std::size_t j = 0; j < nSpacePoints[1]; ++j)
							ASSERT_FLOAT_EQ(solution[GetIndex(i, j, k)], inputDataFloat.initialCondition[GetIndex(i, j, k)]) << static_cast<int>(solver);
			}
		}

		// test double precision
		{
			pde::LinearOperatorProblem<double> problem(inputDataDouble);
			for (size_t n = 0; n < 10; ++n)
			{
				problem.Advance(solver);
				const auto& solution = problem.GetSolution();
				for (std::size_t k = 0; k < nSpacePoints[2]; ++k)
					for (std::size_t i = 0; i < nSpacePoints[0]; ++i)
						for (std::size_t j = 0; j < nSpacePoints[1]; ++j)
							ASSERT_DOUBLE_EQ(solution[GetIndex(i, j, k)], inputDataDouble.initialCondition[GetIndex(i, j, k)]) << static_cast<int>(solver);
			}
		}
	}
}

TEST_F(TransportEquationTestsWithLinearOpeator, ConstantSolution)
{
	for (auto solver : { pde::SolverType::ExplicitEuler, pde::SolverType::LaxWendroff, pde::SolverType::ImplicitEuler, pde::SolverType::CrankNicolson })
	{
		// test single precision
		{
			std::vector<float> nonZeroVelocity(totalSize, 1.0f);
			inputDataFloat.velocityField.fill(nonZeroVelocity);
			std::fill(inputDataFloat.initialCondition.begin(), inputDataFloat.initialCondition.end(), 1.0);
			pde::LinearOperatorProblem<float> problem(inputDataFloat);
			for (size_t n = 0; n < 10; ++n)
			{
				problem.Advance(solver);
				const auto& solution = problem.GetSolution();
				for (std::size_t k = 0; k < nSpacePoints[2]; ++k)
					for (std::size_t i = 0; i < nSpacePoints[0]; ++i)
						for (std::size_t j = 0; j < nSpacePoints[1]; ++j)
						{
							float tol = 0.0;
							switch (solver)
							{
								case pde::SolverType::ExplicitEuler:
								case pde::SolverType::LaxWendroff:
									tol = 5.4e-5f;
									break;
								case pde::SolverType::ImplicitEuler:
									tol = 1.5e-3f;
									break;
								case pde::SolverType::CrankNicolson:
									// numerical instabilities for this is too big! Use doubles!
									continue;
								default:
									ASSERT_FALSE(true);
									break;
							}
							EXPECT_NEAR(solution[GetIndex(i, j, k)], inputDataFloat.initialCondition[GetIndex(i, j, k)], tol) << static_cast<int>(solver);
						}
			}
		}

		// test double precision
		{
			std::vector<double> nonZeroVelocity(totalSize, 1.0);
			inputDataDouble.velocityField.fill(nonZeroVelocity);
			std::fill(inputDataDouble.initialCondition.begin(), inputDataDouble.initialCondition.end(), 1.0);
			pde::LinearOperatorProblem<double> problem(inputDataDouble);
			for (size_t n = 0; n < 10; ++n)
			{
				problem.Advance(solver);
				const auto& solution = problem.GetSolution();
				for (std::size_t k = 0; k < nSpacePoints[2]; ++k)
					for (std::size_t i = 0; i < nSpacePoints[0]; ++i)
						for (std::size_t j = 0; j < nSpacePoints[1]; ++j)
						{
							double tol = 0.0;
							switch (solver)
							{
								case pde::SolverType::ExplicitEuler:
								case pde::SolverType::LaxWendroff:
									tol = 9.4e-14;
									break;
								case pde::SolverType::ImplicitEuler:
									tol = 5e-12;
									break;
								case pde::SolverType::CrankNicolson:
									tol = 2e-10;
									break;
								default:
									ASSERT_FALSE(true);
									break;
							}
							EXPECT_NEAR(solution[GetIndex(i, j, k)], inputDataDouble.initialCondition[GetIndex(i, j, k)], tol) << static_cast<int>(solver);
						}
			}
		}
	}
}

TEST_F(TransportEquationTestsWithLinearOpeator, TransportAlongAxes)
{
	for (auto solver : { pde::SolverType::ExplicitEuler, pde::SolverType::LaxWendroff, pde::SolverType::ImplicitEuler, pde::SolverType::CrankNicolson })
	{
		// test single precision
		{
			for (size_t axisIdx = 0; axisIdx < 3; ++axisIdx)
			{
				// NB: the error is proportional to v * dt
				inputDataFloat.deltaTime = 1e-2f;
				const auto velocity = 0.1f;
				inputDataFloat.velocityField.fill(std::vector<float>(totalSize, 0.0));
				inputDataFloat.velocityField[axisIdx] = std::vector<float>(totalSize, velocity);

				pde::LinearOperatorProblem<float> problem(inputDataFloat);
				for (size_t n = 0; n < 20; ++n)
				{
					problem.Advance(solver);
					const auto& solution = problem.GetSolution();
					for (std::size_t k = 1; k < nSpacePoints[2] - 1; ++k)
					{
						for (std::size_t i = 1; i < nSpacePoints[0] - 1; ++i)
						{
							for (std::size_t j = 1; j < nSpacePoints[1] - 1; ++j)
							{
								ASSERT_NE(solution[GetIndex(i, j, k)], inputDataFloat.initialCondition[GetIndex(i, j, k)]);

								if (axisIdx == 0)
								{
									const auto characteristicX = inputDataFloat.spaceGrids[0][i] + velocity * inputDataFloat.deltaTime * static_cast<float>(n + 1);
									const auto expected = InitialCondition(characteristicX, inputDataFloat.spaceGrids[1][j], inputDataFloat.spaceGrids[2][k]);
									const auto actual = solution[GetIndex(i, j, k)];
									ASSERT_NEAR(actual, expected, 6.55e-3) << n << " | " << i << " | " << j << " | " << k << " --- " << static_cast<int>(solver);
								}
								else if (axisIdx == 1)
								{
									const auto characteristicY = inputDataFloat.spaceGrids[1][j] + velocity * inputDataFloat.deltaTime * static_cast<float>(n + 1);
									const auto expected = InitialCondition(inputDataFloat.spaceGrids[0][i], characteristicY, inputDataFloat.spaceGrids[2][k]);
									const auto actual = solution[GetIndex(i, j, k)];
									ASSERT_NEAR(actual, expected, 6.55e-3) << n << " | " << i << " | " << j << " | " << k;
								}
								else if (axisIdx == 2)
								{
									const auto characteristicZ = inputDataFloat.spaceGrids[2][k] + velocity * inputDataFloat.deltaTime * static_cast<float>(n + 1);
									const auto expected = InitialCondition(inputDataFloat.spaceGrids[0][i], inputDataFloat.spaceGrids[1][j], characteristicZ);
									const auto actual = solution[GetIndex(i, j, k)];
									ASSERT_NEAR(actual, expected, 6.55e-3) << n << " | " << i << " | " << j << " | " << k;
								}
							}
						}
					}
				}
			}
		}

		// test double precision
		{
			for (size_t axisIdx = 0; axisIdx < 3; ++axisIdx)
			{
				// NB: the error is proportional to v * dt
				inputDataDouble.deltaTime = 1e-2;
				const auto velocity = 0.1;
				inputDataDouble.velocityField.fill(std::vector<double>(totalSize, 0.0));
				inputDataDouble.velocityField[axisIdx] = std::vector<double>(totalSize, velocity);

				pde::LinearOperatorProblem<double> problem(inputDataDouble);
				for (size_t n = 0; n < 20; ++n)
				{
					problem.Advance(solver);
					const auto& solution = problem.GetSolution();
					for (std::size_t k = 1; k < nSpacePoints[2] - 1; ++k)
					{
						for (std::size_t i = 1; i < nSpacePoints[0] - 1; ++i)
						{
							for (std::size_t j = 1; j < nSpacePoints[1] - 1; ++j)
							{
								ASSERT_NE(solution[GetIndex(i, j, k)], inputDataDouble.initialCondition[GetIndex(i, j, k)]);

								if (axisIdx == 0)
								{
									const auto characteristicX = inputDataDouble.spaceGrids[0][i] + velocity * inputDataDouble.deltaTime * static_cast<double>(n + 1);
									const auto expected = InitialCondition(characteristicX, inputDataDouble.spaceGrids[1][j], inputDataDouble.spaceGrids[2][k]);
									const auto actual = solution[GetIndex(i, j, k)];
									EXPECT_NEAR(actual, expected, 6.55e-3) << n << " | " << i << " | " << j << " | " << k;
								}
								else if (axisIdx == 1)
								{
									const auto characteristicY = inputDataDouble.spaceGrids[1][j] + velocity * inputDataDouble.deltaTime * static_cast<double>(n + 1);
									const auto expected = InitialCondition(inputDataDouble.spaceGrids[0][i], characteristicY, inputDataDouble.spaceGrids[2][k]);
									const auto actual = solution[GetIndex(i, j, k)];
									ASSERT_NEAR(actual, expected, 6.55e-3) << n << " | " << i << " | " << j << " | " << k;
								}
								else if (axisIdx == 2)
								{
									const auto characteristicZ = inputDataDouble.spaceGrids[2][k] + velocity * inputDataDouble.deltaTime * static_cast<double>(n + 1);
									const auto expected = InitialCondition(inputDataDouble.spaceGrids[0][i], inputDataDouble.spaceGrids[1][j], characteristicZ);
									const auto actual = solution[GetIndex(i, j, k)];
									ASSERT_NEAR(actual, expected, 6.55e-3) << n << " | " << i << " | " << j << " | " << k;
								}
							}
						}
					}
				}
			}
		}
	}
}
