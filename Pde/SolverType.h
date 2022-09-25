
#pragma once

#include <ostream>
#include <cassert>

namespace pde
{
	enum class SolverType
	{
		Null,
		ExplicitEuler,
		ImplicitEuler,
		CrankNicolson,
		LaxWendroff,
	};
}
