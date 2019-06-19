#include "Catch2/single_include/catch2/catch.hpp"

#include "../tinydiff.h"

using namespace tinydiff;

static void RequireNdArray(const NdArray& m, const std::string& str) {
    std::stringstream ss;
    ss << m;
    REQUIRE(ss.str() == str);
}

static bool IsSameNdArray(const NdArray& m1, const NdArray& m2) {
    if (m1.shape() != m2.shape()) {
        return false;
    }
    const float* data1 = m1.data();
    const float* data2 = m2.data();
    for (size_t i = 0; i < m1.size(); i++) {
        if (*(data1++) != *(data2++)) {
            return false;
        }
    }
    return true;
}

TEST_CASE("NdArray") {
    SECTION("Empty") {
        const NdArray m1;
        REQUIRE(m1.empty());
        REQUIRE(m1.size() == 0);
        REQUIRE(m1.shape() == Shape{0});
    }

    SECTION("Float initializer") {
        const NdArray m1 = {1.f, 2.f, 3.f};
        const NdArray m2 = {{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}};
        const NdArray m3 = {{{1.f, 2.f}}, {{3.f, 4.f}}, {{2.f, 3.f}}};
        REQUIRE(m1.shape() == Shape{3});
        REQUIRE(m2.shape() == Shape{2, 3});
        REQUIRE(m3.shape() == Shape{3, 1, 2});
        RequireNdArray(m1, "[1, 2, 3]");
        RequireNdArray(m2,
                       "[[1, 2, 3],\n"
                       " [4, 5, 6]]");
        RequireNdArray(m3,
                       "[[[1, 2]],\n"
                       " [[3, 4]],\n"
                       " [[2, 3]]]");
    }

    SECTION("Float initializer invalid") {
        REQUIRE_NOTHROW(NdArray{{{1.f, 2.f}}, {{3.f, 4.f}}, {{1.f, 2.f}}});
        REQUIRE_THROWS(NdArray{{{1, 2}}, {}});
        REQUIRE_THROWS(NdArray{{{1.f, 2.f}}, {{3.f, 4.f}}, {{1.f, 2.f, 3.f}}});
    }

    SECTION("Confusable initializers") {
        const NdArray m1 = {1.f, 2.f, 3.f};  // Float initializer
        const NdArray m2 = {1, 2, 3};        // Shape (int) initalizer
        const NdArray m3 = {{1, 2, 3}};      // Float initializer due to nest
        REQUIRE(m1.shape() == Shape{3});
        REQUIRE(m2.shape() == Shape{1, 2, 3});
        REQUIRE(m3.shape() == Shape{1, 3});
    }

    SECTION("Empty/Ones/Zeros") {
        const NdArray m1({2, 5});  // Same as Empty
        const auto m2 = NdArray::Empty({2, 5});
        const auto m3 = NdArray::Zeros({2, 5});
        const auto m4 = NdArray::Ones({2, 5});
        REQUIRE(m1.shape() == Shape{2, 5});
        REQUIRE(m2.shape() == Shape{2, 5});
        REQUIRE(m3.shape() == Shape{2, 5});
        REQUIRE(m4.shape() == Shape{2, 5});
        RequireNdArray(m3,
                       "[[0, 0, 0, 0, 0],\n"
                       " [0, 0, 0, 0, 0]]");
        RequireNdArray(m4,
                       "[[1, 1, 1, 1, 1],\n"
                       " [1, 1, 1, 1, 1]]");
    }

    SECTION("Empty/Ones/Zeros by template") {
        const NdArray m1({2, 5});  // No template support
        const auto m2 = NdArray::Empty(2, 5);
        const auto m3 = NdArray::Zeros(2, 5);
        const auto m4 = NdArray::Ones(2, 5);
        REQUIRE(m1.shape() == Shape{2, 5});
        REQUIRE(m2.shape() == Shape{2, 5});
        REQUIRE(m3.shape() == Shape{2, 5});
        REQUIRE(m4.shape() == Shape{2, 5});
        RequireNdArray(m3,
                       "[[0, 0, 0, 0, 0],\n"
                       " [0, 0, 0, 0, 0]]");
        RequireNdArray(m4,
                       "[[1, 1, 1, 1, 1],\n"
                       " [1, 1, 1, 1, 1]]");
    }

    SECTION("Arange") {
        const auto m1 = NdArray::Arange(10.f);
        const auto m2 = NdArray::Arange(0.f, 10.f, 1.f);
        const auto m3 = NdArray::Arange(5.f, 5.5f, 0.1f);
        REQUIRE(m1.shape() == Shape{10});
        REQUIRE(m2.shape() == Shape{10});
        REQUIRE(m3.shape() == Shape{5});
        RequireNdArray(m1, "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]");
        RequireNdArray(m2, "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]");
        RequireNdArray(m3, "[5, 5.1, 5.2, 5.3, 5.4]");
    }

    SECTION("Random uniform") {
        NdArray::Seed(0);
        const auto m1 = NdArray::Uniform({2, 3});
        NdArray::Seed(0);
        const auto m2 = NdArray::Uniform({2, 3});
        NdArray::Seed(1);
        const auto m3 = NdArray::Uniform({2, 3});
        REQUIRE(IsSameNdArray(m1, m2));
        REQUIRE(!IsSameNdArray(m1, m3));
    }

    SECTION("Random normal") {
        NdArray::Seed(0);
        const auto m1 = NdArray::Normal({2, 3});
        NdArray::Seed(0);
        const auto m2 = NdArray::Normal({2, 3});
        NdArray::Seed(1);
        const auto m3 = NdArray::Normal({2, 3});
        REQUIRE(IsSameNdArray(m1, m2));
        REQUIRE(!IsSameNdArray(m1, m3));
    }

    SECTION("Begin/End") {
        auto m1 = NdArray::Arange(1.f, 10.01f);
        // C++11 for-loop
        float sum1 = 0.f;
        for (auto&& v : m1) {
            sum1 += v;
        }
        REQUIRE(sum1 == Approx(55.f));
        // std library
        float sum2 = std::accumulate(m1.begin(), m1.end(), 0.f);
        REQUIRE(sum2 == Approx(55.f));
    }

    SECTION("Float cast") {
        auto m1 = NdArray::Ones({1, 1});
        auto m2 = NdArray::Ones({1, 2});
        REQUIRE(static_cast<float>(m1) == 1);
        REQUIRE_THROWS(static_cast<float>(m2));
    }

    SECTION("Index access by []") {
        auto m1 = NdArray::Arange(12.f);
        auto m2 = NdArray::Arange(12.f).reshape({3, 4});
        auto m3 = NdArray::Arange(12.f).reshape({2, 2, -1});
        m1[3] = -1.f;
        m1[-2] = -2.f;
        m2[{1, 1}] = -1.f;
        m2[{-1, 3}] = -2.f;
        m3[{1, 1, 2}] = -1.f;
        m3[{0, 1, -2}] = -2.f;
        RequireNdArray(m1, "[0, 1, 2, -1, 4, 5, 6, 7, 8, 9, -2, 11]");
        RequireNdArray(m2,
                       "[[0, 1, 2, 3],\n"
                       " [4, -1, 6, 7],\n"
                       " [8, 9, 10, -2]]");
        RequireNdArray(m3,
                       "[[[0, 1, 2],\n"
                       "  [3, -2, 5]],\n"
                       " [[6, 7, 8],\n"
                       "  [9, 10, -1]]]");
    }

    SECTION("Index access by ()") {
        auto m1 = NdArray::Arange(12.f);
        auto m2 = NdArray::Arange(12.f).reshape({3, 4});
        auto m3 = NdArray::Arange(12.f).reshape({2, 2, -1});
        m1(3) = -1.f;
        m1(-2) = -2.f;
        m2(1, 1) = -1.f;
        m2(-1, 3) = -2.f;
        m3(1, 1, 2) = -1.f;
        m3(0, 1, -2) = -2.f;
        RequireNdArray(m1, "[0, 1, 2, -1, 4, 5, 6, 7, 8, 9, -2, 11]");
        RequireNdArray(m2,
                       "[[0, 1, 2, 3],\n"
                       " [4, -1, 6, 7],\n"
                       " [8, 9, 10, -2]]");
        RequireNdArray(m3,
                       "[[[0, 1, 2],\n"
                       "  [3, -2, 5]],\n"
                       " [[6, 7, 8],\n"
                       "  [9, 10, -1]]]");
    }

    SECTION("Index access by [] (const)") {
        const auto m1 = NdArray::Arange(12.f).reshape({1, 4, 3});
        REQUIRE(m1[{0, 2, 1}] == 7);
        REQUIRE(m1[{0, -1, 1}] == 10);
    }

    SECTION("Index access by () (const)") {
        const auto m1 = NdArray::Arange(12.f).reshape({1, 4, 3});
        REQUIRE(m1(0, 2, 1) == 7);
        REQUIRE(m1(0, -1, 1) == 10);
    }

    SECTION("Reshape") {
        auto m1 = NdArray::Arange(12.f);
        auto m2 = m1.reshape({3, 4});
        auto m3 = m2.reshape({2, -1});
        auto m4 = m3.reshape({2, 2, -1});
        RequireNdArray(m1, "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]");
        RequireNdArray(m2,
                       "[[0, 1, 2, 3],\n"
                       " [4, 5, 6, 7],\n"
                       " [8, 9, 10, 11]]");
        RequireNdArray(m3,
                       "[[0, 1, 2, 3, 4, 5],\n"
                       " [6, 7, 8, 9, 10, 11]]");
        RequireNdArray(m4,
                       "[[[0, 1, 2],\n"
                       "  [3, 4, 5]],\n"
                       " [[6, 7, 8],\n"
                       "  [9, 10, 11]]]");
    }

    SECTION("Reshape by template") {
        auto m1 = NdArray::Arange(12.f);
        auto m2 = m1.reshape(3, 4);
        auto m3 = m2.reshape(2, -1);
        auto m4 = m3.reshape(2, 2, -1);
        RequireNdArray(m1, "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]");
        RequireNdArray(m2,
                       "[[0, 1, 2, 3],\n"
                       " [4, 5, 6, 7],\n"
                       " [8, 9, 10, 11]]");
        RequireNdArray(m3,
                       "[[0, 1, 2, 3, 4, 5],\n"
                       " [6, 7, 8, 9, 10, 11]]");
        RequireNdArray(m4,
                       "[[[0, 1, 2],\n"
                       "  [3, 4, 5]],\n"
                       " [[6, 7, 8],\n"
                       "  [9, 10, 11]]]");
    }

    SECTION("Reshape with value change") {
        auto m1 = NdArray::Arange(12.f);
        auto m2 = m1.reshape({3, 4});
        auto m3 = m2.reshape({2, -1});
        auto m4 = m3.reshape({2, 2, -1});
        m1.data()[0] = -1.f;
        RequireNdArray(m1, "[-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]");
        RequireNdArray(m2,
                       "[[-1, 1, 2, 3],\n"
                       " [4, 5, 6, 7],\n"
                       " [8, 9, 10, 11]]");
        RequireNdArray(m3,
                       "[[-1, 1, 2, 3, 4, 5],\n"
                       " [6, 7, 8, 9, 10, 11]]");
        RequireNdArray(m4,
                       "[[[-1, 1, 2],\n"
                       "  [3, 4, 5]],\n"
                       " [[6, 7, 8],\n"
                       "  [9, 10, 11]]]");
    }

    SECTION("Reshape invalid") {
        auto m1 = NdArray::Arange(12.f);
        REQUIRE_THROWS(m1.reshape({5, 2}));
        REQUIRE_THROWS(m1.reshape({-1, -1}));
    }

    SECTION("Slice 2-dim") {
        auto m1 = NdArray::Arange(16.f).reshape(4, 4);
        auto m2 = m1.slice({{1, 3}, {1, 3}});
        auto m3 = m1.slice({1, 3}, {0, 4});
        auto m4 = m1.slice({0, 4}, {1, 3});
        auto m5 = m1.slice({1, -1}, {0, 100000});
        REQUIRE(m1.shape() == Shape{4, 4});
        REQUIRE(m2.shape() == Shape{2, 2});
        REQUIRE(m3.shape() == Shape{2, 4});
        REQUIRE(m4.shape() == Shape{4, 2});
        REQUIRE(m5.shape() == Shape{2, 4});
        RequireNdArray(m1,
                       "[[0, 1, 2, 3],\n"
                       " [4, 5, 6, 7],\n"
                       " [8, 9, 10, 11],\n"
                       " [12, 13, 14, 15]]");
        RequireNdArray(m2,
                       "[[5, 6],\n"
                       " [9, 10]]");
        RequireNdArray(m3,
                       "[[4, 5, 6, 7],\n"
                       " [8, 9, 10, 11]]");
        RequireNdArray(m4,
                       "[[1, 2],\n"
                       " [5, 6],\n"
                       " [9, 10],\n"
                       " [13, 14]]");
        RequireNdArray(m5,
                       "[[4, 5, 6, 7],\n"
                       " [8, 9, 10, 11]]");
    }

    SECTION("Slice high-dim") {
        auto m1 = NdArray::Arange(256).reshape(4, 4, 4, 4);
        auto m2 = m1.slice({{1, 3}, {1, 3}, {1, 3}, {1, 3}});
        auto m3 = m1.slice({1, 3}, {1, 3}, {1, 3}, {1, 3});
        REQUIRE(m1.shape() == Shape{4, 4, 4, 4});
        REQUIRE(m2.shape() == Shape{2, 2, 2, 2});
        REQUIRE(m3.shape() == Shape{2, 2, 2, 2});
        RequireNdArray(m2,
                       "[[[[85, 86],\n"
                       "   [89, 90]],\n"
                       "  [[101, 102],\n"
                       "   [105, 106]]],\n"
                       " [[[149, 150],\n"
                       "   [153, 154]],\n"
                       "  [[165, 166],\n"
                       "   [169, 170]]]]");
        RequireNdArray(m3,
                       "[[[[85, 86],\n"
                       "   [89, 90]],\n"
                       "  [[101, 102],\n"
                       "   [105, 106]]],\n"
                       " [[[149, 150],\n"
                       "   [153, 154]],\n"
                       "  [[165, 166],\n"
                       "   [169, 170]]]]");
    }

    SECTION("Add same shape") {
        auto m1 = NdArray::Arange(12).reshape(2, 3, 2);
        auto m2 = NdArray::Ones({2, 3, 2});
        auto m3 = m1 + m2;
        REQUIRE(m1.shape() == m2.shape());
        REQUIRE(m1.shape() == m3.shape());
        RequireNdArray(m3,
                       "[[[1, 2],\n"
                       "  [3, 4],\n"
                       "  [5, 6]],\n"
                       " [[7, 8],\n"
                       "  [9, 10],\n"
                       "  [11, 12]]]");
    }

    SECTION("Add broadcast 2-dim") {
        auto m1 = NdArray::Arange(6).reshape(2, 3);
        auto m2 = NdArray::Arange(2).reshape(2, 1);
        auto m3 = NdArray::Arange(3).reshape(1, 3);
        auto m12 = m1 + m2;
        auto m13 = m1 + m3;
        auto m23 = m2 + m3;
        REQUIRE(m12.shape() == Shape{2, 3});
        REQUIRE(m13.shape() == Shape{2, 3});
        REQUIRE(m23.shape() == Shape{2, 3});
        RequireNdArray(m12,
                       "[[0, 1, 2],\n"
                       " [4, 5, 6]]");
        RequireNdArray(m13,
                       "[[0, 2, 4],\n"
                       " [3, 5, 7]]");
        RequireNdArray(m23,
                       "[[0, 1, 2],\n"
                       " [1, 2, 3]]");
    }

    SECTION("Add broadcast high-dim") {
        auto m1 = NdArray::Arange(6).reshape(1, 2, 1, 1, 3);
        auto m2 = NdArray::Arange(2).reshape(2, 1);
        auto m3 = NdArray::Arange(3).reshape(1, 3);
        auto m12 = m1 + m2;
        auto m13 = m1 + m3;
        REQUIRE(m12.shape() == Shape{1, 2, 1, 2, 3});
        REQUIRE(m13.shape() == Shape{1, 2, 1, 1, 3});
        RequireNdArray(m12,
                       "[[[[[0, 1, 2],\n"
                       "    [1, 2, 3]]],\n"
                       "  [[[3, 4, 5],\n"
                       "    [4, 5, 6]]]]]");
        RequireNdArray(m13,
                       "[[[[[0, 2, 4]]],\n"
                       "  [[[3, 5, 7]]]]]");
    }

    SECTION("Sub/Mul/Div") {
        auto m1 = NdArray::Arange(6).reshape(2, 3);
        auto m2 = NdArray::Arange(3).reshape(3);
        auto m_sub = m1 - m2;
        auto m_mul = m1 * m2;
        auto m_div = m1 / m2;
        REQUIRE(m_sub.shape() == Shape{2, 3});
        REQUIRE(m_mul.shape() == Shape{2, 3});
        REQUIRE(m_div.shape() == Shape{2, 3});
        RequireNdArray(m_sub,
                       "[[0, 0, 0],\n"
                       " [3, 3, 3]]");
        RequireNdArray(m_mul,
                       "[[0, 1, 4],\n"
                       " [0, 4, 10]]");
        // `0.f / 0.f` can be both of `nan` and `-nan`.
        m_div(0, 0) = std::abs(m_div(0, 0));
        RequireNdArray(m_div,
                       "[[nan, 1, 1],\n"
                       " [inf, 4, 2.5]]");
    }

    SECTION("Arithmetic operators (ndarray, float)") {
        auto m1 = NdArray::Arange(6).reshape(2, 3);
        auto m_add = m1 + 10.f;
        auto m_sub = m1 - 10.f;
        auto m_mul = m1 * 10.f;
        auto m_div = m1 / 10.f;
        RequireNdArray(m_add,
                       "[[10, 11, 12],\n"
                       " [13, 14, 15]]");
        RequireNdArray(m_sub,
                       "[[-10, -9, -8],\n"
                       " [-7, -6, -5]]");
        RequireNdArray(m_mul,
                       "[[0, 10, 20],\n"
                       " [30, 40, 50]]");
        RequireNdArray(m_div,
                       "[[0, 0.1, 0.2],\n"
                       " [0.3, 0.4, 0.5]]");
    }

    SECTION("Arithmetic operators (float, ndarray)") {
        auto m1 = NdArray::Arange(6).reshape(2, 3);
        auto m_add = 10.f + m1;
        auto m_sub = 10.f - m1;
        auto m_mul = 10.f * m1;
        auto m_div = 10.f / m1;
        RequireNdArray(m_add,
                       "[[10, 11, 12],\n"
                       " [13, 14, 15]]");
        RequireNdArray(m_sub,
                       "[[10, 9, 8],\n"
                       " [7, 6, 5]]");
        RequireNdArray(m_mul,
                       "[[0, 10, 20],\n"
                       " [30, 40, 50]]");
        RequireNdArray(m_div,
                       "[[inf, 10, 5],\n"
                       " [3.33333, 2.5, 2]]");
    }

    SECTION("Single +- operators") {
        auto m1 = NdArray::Arange(6).reshape(2, 3);
        auto m_p = +m1;
        auto m_n = -m1;
        RequireNdArray(m_p,
                       "[[0, 1, 2],\n"
                       " [3, 4, 5]]");
        RequireNdArray(m_n,
                       "[[-0, -1, -2],\n"
                       " [-3, -4, -5]]");
    }
}

TEST_CASE("AutoGrad") {
    SECTION("Basic") {
        Variable a(10.f);
        Variable b(20.f);

        Variable c, d, e;
        {
            Variable a2 = F::exp(a);
            c = a2 + b;
            d = c * b;
            e = d * a;
        }

        REQUIRE(a.data() == Approx(10.f));
        REQUIRE(b.data() == Approx(20.f));
        REQUIRE(c.data() == Approx(22046.5f));
        REQUIRE(d.data() == Approx(440929.f));
        REQUIRE(e.data() == Approx(4.40929e+06f));

        e.backward();

        REQUIRE(a.grad() == Approx(4.84622e+06f));
        REQUIRE(b.grad() == Approx(220665.f));
        REQUIRE(c.grad() == Approx(200.f));
        REQUIRE(d.grad() == Approx(10.f));
        REQUIRE(e.grad() == Approx(1.f));
    }
}
