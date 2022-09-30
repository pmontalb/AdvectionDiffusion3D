
#pragma once

#include <array>

namespace pde
{
	[[nodiscard]] static constexpr std::size_t GetIndex(const std::size_t i, const std::size_t j, const std::size_t k, const std::array<std::size_t, 3>& nSpacePoints) noexcept
	{
		return nSpacePoints[2] * (i * nSpacePoints[1] + j) + k;
	}
}	 // namespace pde
