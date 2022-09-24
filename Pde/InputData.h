
#pragma once

#include <vector>
#include <array>

namespace pde
{
	template<typename Real>
	struct InputData
	{
		InputData() = default;
		InputData(const InputData&) = delete;

		std::array<std::vector<Real>, 3> spaceGrids {};
		Real deltaTime {};

		std::vector<Real> initialCondition {};

		std::array<std::vector<Real>, 3> velocityField {};
		std::array<Real, 3> diffusionCoefficients {};
	};
}
