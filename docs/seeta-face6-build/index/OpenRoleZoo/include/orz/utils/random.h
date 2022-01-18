//
// Created by kier on 2018/7/19.
//

#ifndef ORZ_UTILS_RANDOM_H
#define ORZ_UTILS_RANDOM_H

namespace orz {
    class MT19937 {
    public:
        MT19937();

        explicit MT19937(int __seed);

        void srand(int __seed);

        int rand();

        static const int MAX = 0x7fffffff;     // 2 ^ 31 - 1
    private:
        static const int N = 624;      //624 * 32 - 31 = 19937
        int MT[N];
        int m_i = 0;
        int m_seed;
    };

    class Random {
    public:

        Random();

        explicit Random(int __seed);

        // Set seed of random serial
        void seed(int __seed);

        // Uniformly distributed integer in [0, MT19937::MAX]
        int next();

        // Uniformly distributed integer in [min, max]
        int next(int min, int max);

        // Uniformly distributed number in [0, 1]
        double u();

        // Binomial distributed boolean(p)
        bool binomial(double p);

        // Exponential distribution
        double exp(double beta);

        // Ray distribution
        double ray(double mu);

        // Weibull distribution
        double weibull(double alpha, double beta);

        // Normal distribution: N(0, 1)
        double normal();

        // Normal distribution: N(mu, delta^2)
        double normal(double mu, double delta);

    private:
        MT19937 mt;
    };

    extern Random random;
}


#endif //ORZ_UTILS_RANDOM_H
