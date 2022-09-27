
#pragma once

namespace pde
{
	template<typename Real>
	class ISpaceDiscretizer
	{
	public:
		virtual ~ISpaceDiscretizer() = default;
		virtual bool Compute() noexcept = 0;
	};
}	 // namespace pde
