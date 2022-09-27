
#pragma once

#include <array>

namespace pde
{
	[[nodiscard]] static constexpr std::size_t GetIndex(const std::size_t i, const std::size_t j, const std::size_t k, const std::array<std::size_t, 3>& nSpacePoints) noexcept
	{
		return k * nSpacePoints[0] * nSpacePoints[1] + i * nSpacePoints[0] + j;
	}
}	 // namespace pde
