#include "Catch2/single_include/catch2/catch.hpp"

#include "../tinydiff.h"

#include <iomanip>

using namespace tinydiff;

static void CheckNdArray(const Variable& v, const std::string& str,
                         int precision = -1) {
    std::stringstream ss;
    if (0 < precision) {
        ss << std::setprecision(4);
    }
    ss << v;
    CHECK(ss.str() == str);
}

static void CheckData(const Variable& v, const std::string& str,
                      int precision = -1) {
    CheckNdArray(v.data(), str, precision);
}

static void CheckGrad(const Variable& v, const std::string& str,
                      int precision = -1) {
    CheckNdArray(v.grad(), str, precision);
}

TEST_CASE("AutoGrad") {
    // -------------------------- Basic construction ---------------------------
    SECTION("Basic") {
        const Variable v0;
        const Variable v1 = {1.f, 2.f};
        const Variable v2 = {{3.f, 4.f}, {5.f, 6.f}};
        const Variable v3(NdArray::Zeros(2, 4));
        CHECK(v0.data().empty());
        CheckData(v1, "[1, 2]");
        CheckData(v2,
                  "[[3, 4],\n"
                  " [5, 6]]");
        CheckData(v3,
                  "[[0, 0, 0, 0],\n"
                  " [0, 0, 0, 0]]");
    }

    // ------------------------- Arithmetic functions --------------------------
    SECTION("Arithmetic") {
        Variable v1 = {1.f, 2.f};
        Variable v2 = {{3.f, 4.f}, {5.f, 6.f}};

        auto v12 = v1 * v2;
        v12.backward();
        CheckGrad(v1, "[8, 10]");
        CheckGrad(v2,
                  "[[1, 2],\n"
                  " [1, 2]]");

        v1.clearGrad();
        v2.clearGrad();

        auto v21 = v2 * v1;
        v21.backward();
        CheckGrad(v1, "[8, 10]");
        CheckGrad(v2,
                  "[[1, 2],\n"
                  " [1, 2]]");

        CheckData(v1, "[1, 2]");
        CheckData(v2,
                  "[[3, 4],\n"
                  " [5, 6]]");
    }

//         Variable c = a * b;
//         c.backward();
//         std::cout << "* result" << std::endl;
//         std::cout << a.grad() << std::endl;
//         std::cout << b.grad() << std::endl;
//         std::cout << c.grad() << std::endl;

//         Variable c, d, e, f;
//         {
//             Variable a2 = F::Exp(v1);
//             c = a2 + v2;
//             d = c * v2;
//             e = d / v1;
//             f = e - v2;
//         }
//
//         f.backward();
//
//         std::cout << v1.grad() << std::endl;
//         std::cout << v2.grad() << std::endl;
//         std::cout << v2.data() << std::endl;
//         std::cout << c.grad() << std::endl;
//         std::cout << d.grad() << std::endl;
//         std::cout << e.grad() << std::endl;
//         std::cout << f.grad() << std::endl;
}
