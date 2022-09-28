
#pragma once


namespace pde
{
	template<typename T>
	struct Tolerance;

	template<>
	struct Tolerance<float>
	{
		static constexpr float value = 1e-7f;
	};

	template<>
	struct Tolerance<double>
	{
		static constexpr double value = 1e-15;
	};
}
