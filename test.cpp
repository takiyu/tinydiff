#include "tinydiff.h"

int main(int argc, char const* argv[]) {
    using namespace tinydiff;

    {
        NdArray a;
        NdArray b({3});
        NdArray c({2, 3});
        NdArray d({2, 3, 4});
        NdArray e({2, 3, 4, 5});
        std::cout << a << std::endl;
        std::cout << b << std::endl;
        std::cout << c << std::endl;
        std::cout << d << std::endl;
        std::cout << e << std::endl;
    }

    {
        auto&& a = NdArray::Zeros({2, 5});
        auto&& b = NdArray::Ones({2, 5});
        std::cout << a << std::endl;
        std::cout << b << std::endl;
    }

    {
        Variable a(10.f);
        Variable b(20.f);

        Variable c, d, e;
        {
            Variable a2 = F::exp(a);
            c = a2 + b;
            d = c * b;
            e = d * a;
        }

        std::cout << a.data() << std::endl;
        std::cout << b.data() << std::endl;
        std::cout << c.data() << std::endl;
        std::cout << d.data() << std::endl;
        std::cout << e.data() << std::endl;

        e.backward();

        std::cout << a.grad() << std::endl;
        std::cout << b.grad() << std::endl;
        std::cout << c.grad() << std::endl;
        std::cout << d.grad() << std::endl;
        std::cout << e.grad() << std::endl;
    }

    return 0;
}
