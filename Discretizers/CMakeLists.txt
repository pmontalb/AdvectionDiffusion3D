
include_guard()

create_library(
	NAME
		SpaceDiscretizers
	SOURCES
		CenteredDifferenceSpaceDiscretizer.cpp
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
	DEPENDENCIES
		SpaceDiscretizers
)
