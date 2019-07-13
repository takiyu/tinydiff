#include "Catch2/single_include/catch2/catch.hpp"

#include "../tinydiff.h"

using namespace tinydiff;

TEST_CASE("AutoGrad") {
    SECTION("Basic") {
        Variable a(NdArray{{1.f, 10.f}, {2.f, 3.f}});
        Variable b({2.f, 20.f});

//         Variable c = a * b;
//         c.backward();
//         std::cout << "* result" << std::endl;
//         std::cout << a.grad() << std::endl;
//         std::cout << b.grad() << std::endl;
//         std::cout << c.grad() << std::endl;

        Variable c, d, e, f;
        {
            Variable a2 = F::exp(a);
            c = a2 + b;
            d = c * b;
            e = d / a;
            f = e - b;
        }

        f.backward();

        std::cout << a.grad() << std::endl;
        std::cout << b.grad() << std::endl;
        std::cout << c.grad() << std::endl;
        std::cout << d.grad() << std::endl;
        std::cout << e.grad() << std::endl;
        std::cout << f.grad() << std::endl;

        //         REQUIRE(a.data() == Approx(10.f));
        //         REQUIRE(b.data() == Approx(20.f));
        //         REQUIRE(c.data() == Approx(22046.5f));
        //         REQUIRE(d.data() == Approx(440929.f));
        //         REQUIRE(e.data() == Approx(4.40929e+06f));
        //
        //         REQUIRE(a.grad() == Approx(4.84622e+06f));
        //         REQUIRE(b.grad() == Approx(220665.f));
        //         REQUIRE(c.grad() == Approx(200.f));
        //         REQUIRE(d.grad() == Approx(10.f));
        //         REQUIRE(e.grad() == Approx(1.f));
    }
}
