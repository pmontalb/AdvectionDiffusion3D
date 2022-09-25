
#pragma once

#include "InputData.h"
#include "SolverType.h"

#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
// #include <Eigen/PardisoSupport>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseQR>

namespace pde
{
	[[nodiscard]] static constexpr std::size_t GetIndex(const std::size_t i, const std::size_t j, const std::size_t k, const std::array<std::size_t, 3>& nSpacePoints) noexcept
	{
		return k * nSpacePoints[0] * nSpacePoints[1] + i * nSpacePoints[0] + j;
	}

	namespace detail
	{
		template<typename Real>
		class Vector: public Eigen::VectorX<Real>
		{
		public:
			[[nodiscard]] auto& operator[](const size_t i) { return Eigen::VectorX<Real>::operator[](static_cast<int>(i)); }
			[[nodiscard]] auto& operator[](const size_t i) const { return Eigen::VectorX<Real>::operator[](static_cast<int>(i)); }
			using Eigen::VectorX<Real>::operator=;
			using Eigen::VectorX<Real>::VectorX;
		};
	}	 // namespace detail

	template<typename Real>
	class Problem
	{
	public:
		Problem(const InputData<Real>& inputData) noexcept;
		virtual ~Problem() = default;

		const auto& GetSolution() const noexcept { return _solution; }

		virtual void Advance(const SolverType solverType) noexcept;

	protected:
		[[nodiscard]] size_t GetIndex(const size_t i, const size_t j, const size_t k) const noexcept { return pde::GetIndex(i, j, k, _nSpacePoints); }

		void SetZeroFluxBoundaryConditions();
		void SetZeroFluxBoundaryConditionsAtObstacles();

	protected:
		const InputData<Real>& _inputData;
		std::array<std::size_t, 3> _nSpacePoints;

		detail::Vector<Real> _solution;
	};

	template<typename Real>
	class LinearOperatorProblem: public Problem<Real>
	{
	public:
		using Problem<Real>::Problem;
		void Advance(const SolverType solverType) noexcept override;

		using Problem<Real>::GetIndex;

	private:
		void MakeSpaceOperator(const SolverType) noexcept;

	private:
		using Problem<Real>::_inputData;
		using Problem<Real>::_solution;
		using Problem<Real>::_nSpacePoints;
		Eigen::VectorX<Real> _cache;
		Eigen::VectorX<Real> _cache2;

		using SparseMatrix = Eigen::SparseMatrix<Real>;
		SparseMatrix _spaceOperator {};
		SparseMatrix _rightTimeOperator {};
		SparseMatrix _leftTimeOperator {};

		//		using SparseSolver = Eigen::PardisoLDLT<SparseMatrix>;
//				using SparseSolver = Eigen::SimplicialLLT<SparseMatrix>;
//				using SparseSolver = Eigen::SimplicialLDLT<SparseMatrix>;
//		using SparseSolver = Eigen::ConjugateGradient<SparseMatrix, Eigen::Lower | Eigen::Upper>;
//		using SparseSolver = Eigen::BiCGSTAB<SparseMatrix>;
		using SparseSolver = Eigen::SparseLU<SparseMatrix>;
//		using SparseSolver = Eigen::SparseQR<SparseMatrix, Eigen::COLAMDOrdering<int>>;
		SparseSolver _sparseSolver;


		SolverType _lastSolverType = SolverType::Null;
	};
}	 // namespace pde

extern template class pde::Problem<float>;
extern template class pde::Problem<double>;

extern template class pde::LinearOperatorProblem<float>;
extern template class pde::LinearOperatorProblem<double>;
