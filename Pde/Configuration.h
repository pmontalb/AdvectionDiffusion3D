
#pragma once

#include "SolverType.h"

#include <array>
#include <vector>

namespace pde
{
	struct Configuration
	{
		[[nodiscard]] constexpr std::size_t GetIndex(const std::size_t i, const std::size_t j, const std::size_t k) const noexcept
		{
			return k * nSpacePoints[0] * nSpacePoints[1] + i * nSpacePoints[0] + j;
		}

		// M = { Mx, My, Mz }
		std::array<std::size_t, 3> nSpacePoints {};

		SolverType solverType = SolverType::LaxWendroff;

		// TODO: obstacle indices
	};
}	 // namespace pde
