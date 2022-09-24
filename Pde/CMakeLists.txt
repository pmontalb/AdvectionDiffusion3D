
include_guard()

create_library(
	NAME
		Pde
	SOURCES
		Problem.cpp
)

create_executable(
	NAME
		pdeDebug
	SOURCES
		Debug.cpp
	DEPENDENCIES
		Pde
)
