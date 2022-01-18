//
// Created by kier on 2018/7/19.
//

#include "orz/utils/random.h"

#include <ctime>
#include <cstdlib>
#include <cmath>

namespace orz {
    MT19937::MT19937() {
        this->srand((int) time(nullptr));
    }

    MT19937::MT19937(int __seed) {
        this->srand(__seed);
    }

    void MT19937::srand(int __seed) {
        this->m_seed = __seed;
        this->m_i = 0;
        MT[0] = __seed;
        for (int i = 1; i < N; i++) {
            MT[i] = 0x6c078965 * (MT[i - 1] ^ (MT[i - 1] >> 30)) + i;
            MT[i] &= 0xffffffff;   // get the last 32bits
        }
    }

    int MT19937::rand() {
        int i = this->m_i;
        int generate = (MT[i] & 0x80000000) + (MT[(i + 1) % 624] & 0x7fffffff);
        MT[i] = MT[(i + 397) % 624] ^ (generate >> 1);
        if (generate & 1) MT[i] ^= 0x9908b0df;

        int y = MT[i];
        y = y ^ (y >> 11);
        y = y ^ ((y << 7) & 0x9d2c5680);
        y = y ^ ((y << 15) & 0xefc60000);
        y = y ^ (y >> 18);
        this->m_i = (i + 1) % 624;
        return y;
    }

    Random::Random()
            : mt() {
    }

    Random::Random(int __seed)
            : mt(__seed) {
    }

    void Random::seed(int __seed) {
        mt.srand(__seed);
    }

    int Random::next() {
        return mt.rand();
    }

    int Random::next(int min, int max) {
        // return min + (int) ((max - min) * u());
        return min + (mt.rand() % (max - min + 1));
    }

    double Random::u() {
        return (double) mt.rand() / MT19937::MAX;
    }

    bool Random::binomial(double p) {
        return u() < p;
    }

#define ln log
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif // M_PI

    double Random::exp(double beta) {
        return -beta * ln(u());
    }

    double Random::ray(double mu) {
        return sqrt(-2 * mu * mu * ln(u()));
    }

    double Random::weibull(double alpha, double beta) {
        return beta * pow(-ln(u()), 1.0 / alpha);
    }

    double Random::normal() {
        return sqrt(-2 * ln(u())) * sin(2 * M_PI * u());
    }

    double Random::normal(double mu, double delta) {
        return mu + delta * normal();
    }

    Random random;
}