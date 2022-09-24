
#pragma once

#include <ostream>
#include <cassert>

namespace pde
{
	enum class SolverType
	{
		ExplicitEuler,
		ImplicitEuler,
		LaxWendroff,
	};
}
