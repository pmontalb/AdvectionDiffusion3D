
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
		laUnitTests
	SOURCES
		TridiagonalSolver.cpp
	DEPENDENCIES
		testMain
	SYSTEM_DEPENDENCIES
		Eigen3::Eigen
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
