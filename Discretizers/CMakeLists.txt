
include_guard()

create_library(
	NAME
		SpaceDiscretizers
	SOURCES
		CenteredDifferenceSpaceDiscretizer.cpp
		AdiSpaceDiscretizer.cpp
	SYSTEM_DEPENDENCIES
		Eigen3::Eigen
#		MKL::MKL
)

create_library(
	NAME
		TimeDiscretizers
	SOURCES
		ExplicitTimeDiscretizer.cpp
		ImplicitTimeDiscretizer.cpp
		CrankNicolsonTimeDiscretizer.cpp
		AdiTimeDiscretizer.cpp
	DEPENDENCIES
		SpaceDiscretizers
)
