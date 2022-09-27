
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

	constexpr std::string_view ToString(const SolverType type)
	{
		switch (type)
		{
			case SolverType::Null:
				return "Null";
			case SolverType::ExplicitEuler:
				return "ExplicitEuler";
			case SolverType::ImplicitEuler:
				return "ImplicitEuler";
			case SolverType::CrankNicolson:
				return "CrankNicolson";
			case SolverType::LaxWendroff:
				return "LaxWendroff";
			default:
				return "?";
		}
	}
}
