// black_scholes_openmp_parallel_opt.cpp
// Explicit finite-difference Black-Scholes solver with improved OpenMP pattern.
// Persistent parallel team, single for boundary/swap, omp for inside.
// Compile: g++ -O3 -march=native -fopenmp BSE_ExplicitParallel_OpenMP.cpp -o bs_explicit_opt
// Run: ./bs_explicit_opt 1

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <omp.h>

using namespace std;

// Standard normal CDF via erf
double normal_cdf(double x) {
    return 0.5 * (1.0 + erf(x / sqrt(2.0)));
}

// Black-Scholes analytic European call (for validation)
double bs_call_price(double S, double K, double r, double sigma, double T, double t = 0.0) {
    double tau = T - t;
    if (tau <= 0.0) return max(S - K, 0.0);
    if (S <= 0.0) return 0.0;
    double sqrt_tau = sqrt(tau);
    double d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * tau) / (sigma * sqrt_tau);
    double d2 = d1 - sigma * sqrt_tau;
    return S * normal_cdf(d1) - K * exp(-r * tau) * normal_cdf(d2);
}

int main(int argc, char* argv[]) {
    // allow user to set number of threads via argv[1]
    if (argc > 1) {
        int t = atoi(argv[1]);
        if (t > 0) omp_set_num_threads(t);
    }

    // Problem parameters (change if needed)
    const double K = 85.0;
    const double T = 1.0;
    const double r = 0.1;
    const double sigma = 0.2;
    const double S0 = 90.0;

    // Grid parameters (tune M and N to match workload)
    const double Smax = 2.0 * K;
    const size_t M = 3200;          // spatial intervals (M+1 nodes)
    const size_t N = 500000;       // time steps

    double dS = Smax / double(M);
    double dt = T / double(N);

    cout << "Black-Scholes explicit FD (optimized OpenMP)" << endl;
    cout << "K=" << K << " T=" << T << " r=" << r << " sigma=" << sigma << endl;
    cout << "Smax=" << Smax << " M=" << M << " dS=" << dS << endl;
    cout << "N=" << N << " dt=" << dt << endl;

    // quick stability indicator
    double stability_ratio = (sigma * sigma) * (Smax * Smax) * dt / (dS * dS);
    cout << "Approx stability ratio = " << stability_ratio << " (should be << 1 for explicit)" << endl;
    if (stability_ratio > 0.5) cout << "WARNING: explicit scheme may be unstable; consider smaller dt or implicit/CN." << endl;

    // build S grid
    vector<double> S(M + 1);
    for (size_t j = 0; j <= M; ++j) S[j] = j * dS;

    // allocate two layers
    vector<double> old_values(M + 1), new_values(M + 1);

    // terminal payoff at t = T
    for (size_t j = 0; j <= M; ++j) old_values[j] = max(S[j] - K, 0.0);

    // performance/timing setup
    int threads_available = omp_get_max_threads();
    cout << "OpenMP threads available (omp_get_max_threads): " << threads_available << endl;
    // read actual threads in use (may be equal to omp_get_max_threads or set by env/argv)
    #pragma omp parallel
    {
        #pragma omp single
        cout << "Running with OMP_NUM_THREADS = " << omp_get_num_threads() << " threads" << endl;
    }

    const double half = 0.5;
    double t_start = omp_get_wtime();

    // Outer parallel region: threads created once and reused
    #pragma omp parallel
    {
        // local references to arrays for tiny speed (no copy)
        // loop over time steps sequentially, but loop over space parallelized
        for (size_t step = 0; step < N; ++step) {
            double t_new = T - double(step + 1) * dt;

            // compute boundary values by a single thread
            #pragma omp single
            {
                new_values[0] = 0.0;
                new_values[M] = Smax - K * exp(-r * (T - t_new));
            }
            // ensure boundaries are set before workers read them
            #pragma omp barrier

            // parallel work across spatial nodes (interior)
            #pragma omp for schedule(static)
            for (int j = 1; j < int(M); ++j) {
                double Sj = S[j];
                // use scaled form to avoid repeated divisions
                double Sj_by_dS = Sj / dS;
                double Sj2_by_dS2 = Sj_by_dS * Sj_by_dS;
                double a = half * dt * (sigma * sigma * Sj2_by_dS2 - r * Sj_by_dS);
                double b = 1.0 - dt * (sigma * sigma * Sj2_by_dS2 + r);
                double c = half * dt * (sigma * sigma * Sj2_by_dS2 + r * Sj_by_dS);

                new_values[j] = a * old_values[j - 1] + b * old_values[j] + c * old_values[j + 1];
            }

            // make sure all threads finished writing new_values before swap
            #pragma omp barrier

            // one thread does the swap (cheap) and then all threads continue
            #pragma omp single
            {
                old_values.swap(new_values);
            }
            // implicit barrier at end of single ensures swap visible to all
        } // end for steps
    } // end parallel

    cout << fixed << setprecision(6);
    double t_end = omp_get_wtime();
    cout << "Time marching took " << (t_end - t_start) << " seconds" << endl;

    // interpolate solution at S0
    double computed_price;
    if (S0 <= 0.0) computed_price = old_values[0];
    else if (S0 >= Smax) computed_price = old_values[M];
    else {
        size_t idx = size_t(S0 / dS);
        if (idx >= M) idx = M - 1;
        double x0 = S[idx], x1 = S[idx + 1];
        double y0 = old_values[idx], y1 = old_values[idx + 1];
        computed_price = y0 + (S0 - x0) * (y1 - y0) / (x1 - x0);
    }

    cout << "Computed explicit FD price at t=0 for S=" << S0 << " : " << computed_price << endl;

    double analytic = bs_call_price(S0, K, r, sigma, T, 0.0);
    cout << "Analytic Black-Scholes price: " << analytic << endl;
    cout << "Absolute error: " << fabs(computed_price - analytic) << endl;

    // write CSV of V(S,0)
    ofstream fout("bs_explicit_solution_t0.csv");
    fout << "S,V\n";
    for (size_t j = 0; j <= M; ++j) fout << S[j] << "," << old_values[j] << "\n";
    fout.close();
    cout << "Wrote bs_explicit_solution_t0.csv" << endl;

    return 0;
}