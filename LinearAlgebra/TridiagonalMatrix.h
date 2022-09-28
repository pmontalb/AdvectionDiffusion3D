
#pragma once

#include <vector>

#include <Eigen/Eigen>

#include <iostream>

namespace la
{
	template<typename Real>
	struct Triple
	{
		Triple() = default;
		Triple(const Real sub_, const Real diag_, const Real super_) noexcept
			: sub(sub_), diag(diag_), super(super_)
		{
		}

		Real sub;
		Real diag;
		Real super;
	};

	template<typename Real>
	class TridiagonalMatrix
	{
	public:
		TridiagonalMatrix() = default;
		TridiagonalMatrix(const std::size_t N)
			: _data(N)
		{
		}
		std::size_t Size() const noexcept { return _data.size(); }

		auto& operator[](const std::size_t i) noexcept { return _data[i]; }
		const auto& operator[](const std::size_t i) const noexcept { return _data[i]; }

		auto& Front() noexcept { return _data.front(); }
		const auto& Front() const noexcept { return _data.front(); }

		auto& Back() noexcept { return _data.back(); }
		const auto& Back() const noexcept { return _data.back(); }

		void Resize(const std::size_t N) noexcept
		{
			_data.resize(N);
		}

		void Push(const double a, const double b, const double c) noexcept
		{
			_data.emplace_back(a, b, c);
		}

		void SetIdentity(const std::size_t N) noexcept
		{
			Resize(N);
			SetIdentity();
		}
		void SetIdentity() noexcept
		{
			for (size_t i = 0; i < _data.size(); ++i)
				_data[i] = { 0.0, 1.0, 0.0 };
		}

		Eigen::VectorX<Real> Dot(const Eigen::VectorX<Real>& in) const noexcept
		{
			Eigen::VectorX<Real> out;
			out.resize(in.size());
			Dot(out, in);

			return out;
		}
		void Dot(Eigen::VectorX<Real>& out, const Eigen::VectorX<Real>& in) const noexcept
		{
			out[0] = _data[0].diag * in[0] + _data[0].super * in[1];
			for (size_t i = 1; i < _data.size() - 1; ++i)
			{
				out[static_cast<int>(i)] = _data[i].sub * in[static_cast<int>(i) - 1];
				out[static_cast<int>(i)] += _data[i].diag * in[static_cast<int>(i)];
				out[static_cast<int>(i)] += _data[i].super * in[static_cast<int>(i) + 1];
			}
			out[static_cast<int>(_data.size() - 1)] = _data[_data.size() - 1].sub * in[static_cast<int>(_data.size() - 2)] + _data[_data.size() - 1].diag * in[static_cast<int>(_data.size() - 1)];
		}

		void Print()
		{
			std::cout << _data.front().diag << ", " << _data.front().super << std::endl;
			for (std::size_t i = 1; i < _data.size() - 1; ++i)
				std::cout << _data[i].sub << ", " << _data[i].diag << ", " << _data[i].super << std::endl;
			std::cout << _data.back().diag << ", " << _data.back().super << std::endl;
		}

	private:
		std::vector<Triple<Real>> _data;
	};
}
