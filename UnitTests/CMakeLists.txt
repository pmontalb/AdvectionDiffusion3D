
include_guard()

create_library(
	NAME
		testMain
	SOURCES
		TestMain.cpp
	SYSTEM_DEPENDENCIES
		gtest
)

create_test(
	NAME
		pdeUnitTests
	SOURCES
		TransportEquation.cpp
		HeatEquation.cpp
		AdvectionDiffusion.cpp
	DEPENDENCIES
		Pde testMain
)
