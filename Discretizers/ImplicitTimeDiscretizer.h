
#pragma once

#include "CenteredDifferenceSpaceDiscretizer.h"
#include "ITimeDiscretizer.h"
#include "Pde/Index.h"
#include "Pde/InputData.h"
#include "Pde/SolverType.h"

//#include <Eigen/Dense>
//#include <Eigen/IterativeLinearSolvers>
//#include <Eigen/PardisoSupport>
#include <Eigen/Sparse>
//#include <Eigen/SparseCholesky>
//#include <Eigen/SparseQR>
//#include <Eigen/LU>
////#include <Eigen/UmfPackSupport>

namespace pde
{
	/*
	 * Implicit Euler
	 * Fully Implicit 7-banded diagonal linear system
	 *
	 * L is the centered difference space discretization operator (7-banded)
	 *
	 * (I - dt * L) * C_{n + 1} = C_n
	 *
	 * +++ Unconditionally stable
	 * --- Unstable if the PDE is advection-dominated
	 * +++ O(dt + dx^2)
	 * --- Prohibitively slow
	 * */
	template<typename Real>
	class ImplicitTimeDiscretizer final: public ITimeDiscretizer<Real>
	{
	public:
		ImplicitTimeDiscretizer(const InputData<Real>& inputData, const CenteredDifferenceSpaceDiscretizer<Real>& spaceDiscretizer) : _inputData(inputData), _spaceDiscretizer(spaceDiscretizer) {}

		bool Precompute() noexcept override;
		void Compute(Eigen::VectorX<Real>& out, const Eigen::VectorX<Real>& in) noexcept override;

	private:
		const InputData<Real>& _inputData;
		const CenteredDifferenceSpaceDiscretizer<Real>& _spaceDiscretizer;

		//		using SparseMatrix = Eigen::SparseMatrix<Real, Eigen::RowMajor>;
		//		using SparseMatrix = Eigen::SparseMatrix<Real, Eigen::ColMajor, long long>;
		using SparseMatrix = Eigen::SparseMatrix<Real>;

		//		using SparseSolver = Eigen::PardisoLU<SparseMatrix>;
		//		using SparseSolver = Eigen::BiCGSTAB<SparseMatrix>;
		using SparseSolver = Eigen::SparseLU<SparseMatrix, Eigen::COLAMDOrdering<int>>;
		//		using SparseSolver = Eigen::SparseQR<SparseMatrix, Eigen::COLAMDOrdering<int>>;
		//		using SparseSolver = Eigen::UmfPackLU<SparseMatrix>;

		SparseSolver _solver {};
		SparseMatrix _operator {};
	};
}	 // namespace pde

extern template class pde::ImplicitTimeDiscretizer<float>;
extern template class pde::ImplicitTimeDiscretizer<double>;
