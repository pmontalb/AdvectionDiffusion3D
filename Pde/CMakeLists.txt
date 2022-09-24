
include_guard()

create_library(
	NAME
		Pde
	SOURCES
		Problem.cpp
	SYSTEM_DEPENDENCIES
		Eigen3::Eigen
)

create_executable(
	NAME
		pdeDebug
	SOURCES
		Debug.cpp
	DEPENDENCIES
		Pde
)
