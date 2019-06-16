#define CATCH_CONFIG_MAIN  // Define main()
#include "Catch2/single_include/catch2/catch.hpp"

#include "../tinydiff.h"

using namespace tinydiff;

static void RequireNdArray(const NdArray& m, const std::string& str) {
    std::stringstream ss;
    ss << m;
    REQUIRE(ss.str() == str);
}

TEST_CASE("NdArray") {
    SECTION("Empty/Ones/Zeros") {
        const NdArray m1({2, 5});
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
