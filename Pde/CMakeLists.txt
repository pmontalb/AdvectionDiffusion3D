
include_guard()

create_library(
	NAME
		Pde
	SOURCES
		Problem.cpp
	DEPENDENCIES
		TimeDiscretizers
)

create_executable(
	NAME
		pdeDebug
	SOURCES
		Debug.cpp
	DEPENDENCIES
		Pde
)

create_executable(
	NAME
		pdeProfiler
	SOURCES
		Profiler.cpp
	DEPENDENCIES
		Pde Profiler
)
