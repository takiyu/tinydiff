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
    SECTION("Basic construction") {
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

    // ----------------------------- Basic methods -----------------------------
    SECTION("Basic methods") {
        const Variable v0;
        const Variable v1 = {1.f, 2.f};
        const Variable v2 = {3.f, 4.f};
        auto v3 = v1;
        auto v12 = v1 - v2;
        CHECK(v0.id() != v1.id());
        CHECK(v1.id() == v3.id());
        CHECK(v0.empty());
        CHECK(!v1.empty());
        CHECK(v0.size() == 0);
        CHECK(v1.size() == 2);
        CHECK(v0.shape() == Shape{0});
        CHECK(v1.shape() == Shape{2});
        CHECK(v0.ndim() == 1);
        CHECK(v1.ndim() == 1);
        v12.backward();
        // Nothing changed after backward
        CHECK(v0.id() != v1.id());
        CHECK(v1.id() == v3.id());
        CHECK(v0.empty());
        CHECK(!v1.empty());
        CHECK(v0.size() == 0);
        CHECK(v1.size() == 2);
        CHECK(v0.shape() == Shape{0});
        CHECK(v1.shape() == Shape{2});
        CHECK(v0.ndim() == 1);
        CHECK(v1.ndim() == 1);
    }

    // ------------------------- Arithmetic functions --------------------------
    SECTION("Arithmetic (add, broadcast)") {
        Variable v1 = {1.f, 2.f};
        Variable v2 = {{3.f, 4.f}, {5.f, 6.f}};
        // Backward From left to right
        auto v12 = v1 + v2;
        v12.backward();
        CheckGrad(v1, "[2, 2]");
        CheckGrad(v2,
                  "[[1, 1],\n"
                  " [1, 1]]");
        // Backward From right to left
        auto v21 = v2 + v1;
        v21.backward();
        CheckGrad(v1, "[2, 2]");
        CheckGrad(v2,
                  "[[1, 1],\n"
                  " [1, 1]]");
        // Data check
        CheckData(v1, "[1, 2]");
        CheckData(v2,
                  "[[3, 4],\n"
                  " [5, 6]]");
    }

    SECTION("Arithmetic (sub, broadcast)") {
        Variable v1 = {1.f, 2.f};
        Variable v2 = {{3.f, 4.f}, {5.f, 6.f}};
        // Backward From left to right
        auto v12 = v1 - v2;
        v12.backward();
        CheckGrad(v1, "[2, 2]");
        CheckGrad(v2,
                  "[[-1, -1],\n"
                  " [-1, -1]]");
        // Backward From right to left
        auto v21 = v2 - v1;
        v21.backward();
        CheckGrad(v1, "[-2, -2]");
        CheckGrad(v2,
                  "[[1, 1],\n"
                  " [1, 1]]");
        // Data check
        CheckData(v1, "[1, 2]");
        CheckData(v2,
                  "[[3, 4],\n"
                  " [5, 6]]");
    }

    SECTION("Arithmetic (mul, broadcast)") {
        Variable v1 = {1.f, 2.f};
        Variable v2 = {{3.f, 4.f}, {5.f, 6.f}};
        // Backward From left to right
        auto v12 = v1 * v2;
        v12.backward();
        CheckGrad(v1, "[8, 10]");
        CheckGrad(v2,
                  "[[1, 2],\n"
                  " [1, 2]]");
        // Backward From right to left
        auto v21 = v2 * v1;
        v21.backward();
        CheckGrad(v1, "[8, 10]");
        CheckGrad(v2,
                  "[[1, 2],\n"
                  " [1, 2]]");
        // Data check
        CheckData(v1, "[1, 2]");
        CheckData(v2,
                  "[[3, 4],\n"
                  " [5, 6]]");
    }

    SECTION("Arithmetic (div, broadcast)") {
        Variable v1 = {1.f, 2.f};
        Variable v2 = {{3.f, 4.f}, {5.f, 6.f}};
        // Backward From left to right
        auto v12 = v1 / v2;
        v12.backward();
        CheckGrad(v1, "[0.533333, 0.416667]");
        CheckGrad(v2,
                  "[[-0.111111, -0.125],\n"
                  " [-0.04, -0.0555556]]");
        // Backward From right to left
        auto v21 = v2 / v1;
        v21.backward();
        CheckGrad(v1, "[-8, -2.5]");
        CheckGrad(v2,
                  "[[1, 0.5],\n"
                  " [1, 0.5]]");
        // Data check
        CheckData(v1, "[1, 2]");
        CheckData(v2,
                  "[[3, 4],\n"
                  " [5, 6]]");
    }

    // -------------------- Arithmetic functions (complex) ---------------------
    SECTION("Arithmetic (complex chain)") {
        Variable v1 = {1.f, 2.f};
        Variable v2 = {{3.f, 4.f}, {5.f, 6.f}};

        auto v3 = v1 + v2;
        auto v4 = v3 * v2;
        auto v5 = v4 / v1;
        auto v6 = v5 - v3;
        v6.backward();

        CheckGrad(v1, "[-36, -15]");
        CheckGrad(v2,
                  "[[6, 4],\n"
                  " [10, 6]]");
    }

    SECTION("Arithmetic (complex chain and shape)") {
        Variable v1(NdArray({1.f, 2.f}).reshape(2, 1));
        Variable v2(NdArray({{3.f, 4.f}, {5.f, 6.f}}).reshape(1, 1, 2, 1, 2));

        auto v3 = v1 * v2;
        auto v4 = v3 - v2;
        auto v5 = v4 + v1;
        auto v6 = v5 / v3;
        v6.backward();

        CheckGrad(v1,
                  "[[4],\n"
                  " [1]]");
        CheckGrad(v2,
                  "[[[[[-0.222222, -0.125]],\n"
                  "   [[-0.08, -0.0555556]]]]]");
    }

    // --------------------------- Backward options ----------------------------
    SECTION("Backward (retain_grads)") {
        Variable v1 = {1.f, 2.f};
        Variable v2 = {{3.f, 4.f}, {5.f, 6.f}};

        // retain_grads: true
        auto v3_a = v1 + v2;
        auto v4_a = v3_a * v2;
        auto v5_a = v3_a / v4_a;
        v5_a.backward(true, true);
        CheckGrad(v1, "[0, 0]");
        CheckGrad(v2,
                  "[[-0.111111, -0.0625],\n"
                  " [-0.04, -0.0277778]]");
        CheckGrad(v3_a,
                  "[[0, 0],\n"
                  " [0, 0]]");
        CheckGrad(v4_a,
                  "[[-0.0277778, -0.0104167],\n"
                  " [-0.00666667, -0.00347222]]");
        CheckGrad(v5_a,
                  "[[1, 1],\n"
                  " [1, 1]]");

        // retain_grads: false
        auto v3_b = v1 + v2;
        auto v4_b = v3_b * v2;
        auto v5_b = v3_b / v4_b;
        v5_b.backward(true, false);
        CheckGrad(v1, "[0, 0]");
        CheckGrad(v2,
                  "[[-0.111111, -0.0625],\n"
                  " [-0.04, -0.0277778]]");
        CheckGrad(v3_b, "[]");
        CheckGrad(v4_b, "[]");
        CheckGrad(v5_b, "[]");
    }

    SECTION("Backward (no clear_grads)") {
        Variable v1 = {1.f, 2.f};
        Variable v2 = {{3.f, 4.f}, {5.f, 6.f}};

        // retain_grads: true
        auto v3_a = v1 + v2;
        auto v4_a = v3_a * v2;
        auto v5_a = v3_a / v4_a;
        v5_a.backward(false, true);
        CheckGrad(v1, "[0, 0]");
        CheckGrad(v2,
                  "[[-0.111111, -0.0625],\n"
                  " [-0.04, -0.0277778]]");
        CheckGrad(v3_a,
                  "[[0, 0],\n"
                  " [0, 0]]");
        CheckGrad(v4_a,
                  "[[-0.0277778, -0.0104167],\n"
                  " [-0.00666667, -0.00347222]]");
        CheckGrad(v5_a,
                  "[[1, 1],\n"
                  " [1, 1]]");

        // retain_grads: true (second)
        auto v3_b = v1 + v2;
        auto v4_b = v3_b * v2;
        auto v5_b = v3_b / v4_b;
        v5_b.backward(false, true);
        CheckGrad(v1, "[0, 0]");
        CheckGrad(v2,
                  "[[-0.222222, -0.125],\n"
                  " [-0.08, -0.0555556]]");
        CheckGrad(v3_b,
                  "[[0, 0],\n"
                  " [0, 0]]");
        CheckGrad(v4_b,
                  "[[-0.0277778, -0.0104167],\n"
                  " [-0.00666667, -0.00347222]]");
        CheckGrad(v5_b,
                  "[[1, 1],\n"
                  " [1, 1]]");
    }
}
