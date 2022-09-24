
#pragma once

#include "Configuration.h"
#include "InputData.h"

#include "Eigen/Dense"
#include "Eigen/Sparse"

namespace pde
{
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
		Problem(const InputData<Real>& inputData, Configuration configuration) noexcept;
		virtual ~Problem() = default;

		const auto& GetSolution() const noexcept { return _solution; }

		virtual void Advance() noexcept;

	protected:
		[[nodiscard]] constexpr size_t GetIndex(const size_t i, const size_t j, const size_t k) const noexcept { return _configuration.GetIndex(i, j, k); }

		void SetZeroFluxBoundaryConditions();
		void SetZeroFluxBoundaryConditionsAtObstacles();

	protected:
		const InputData<Real>& _inputData;
		Configuration _configuration;

		detail::Vector<Real> _solution;
	};

	template<typename Real>
	class LinearOperatorProblem: public Problem<Real>
	{
	public:
		LinearOperatorProblem(const InputData<Real>& inputData, Configuration configuration) noexcept;

		void Advance() noexcept override;

		using Problem<Real>::GetIndex;

	private:
		using Problem<Real>::_inputData;
		using Problem<Real>::_configuration;
		using Problem<Real>::_solution;

		Eigen::SparseMatrix<Real> _spaceOperator {};
		Eigen::SparseMatrix<Real> _timeOperator {};
		Eigen::VectorX<Real> _cache {};
	};
}	 // namespace pde

extern template class pde::Problem<float>;
extern template class pde::Problem<double>;

extern template class pde::LinearOperatorProblem<float>;
extern template class pde::LinearOperatorProblem<double>;
