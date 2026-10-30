#include "pipeline.hpp"
#include "backend_cpu.hpp"
#include "host_ops.hpp"
#include "io.hpp"
#include "solver.hpp"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <unordered_map>
#include <vector>

namespace unmixing {

namespace {

using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::VectorXd;

constexpr int kNumBins = params::kNumberOfBins;

struct Bin3 { int r, g, b; };

Bin3 get_bin(const Vector3d& c) {
    auto f = [](double v) {
        int b = static_cast<int>(std::floor(v * kNumBins));
        if (b < 0) b = 0;
        if (b > kNumBins - 1) b = kNumBins - 1;
        return b;
    };
    return Bin3{ f(c(0)), f(c(1)), f(c(2)) };
}

std::uint32_t quantize_rgb(double r, double g, double b) {
    const std::uint32_t ri = static_cast<std::uint32_t>(
        std::lround(std::clamp(r, 0.0, 1.0) * 255.0));
    const std::uint32_t gi = static_cast<std::uint32_t>(
        std::lround(std::clamp(g, 0.0, 1.0) * 255.0));
    const std::uint32_t bi = static_cast<std::uint32_t>(
        std::lround(std::clamp(b, 0.0, 1.0) * 255.0));
    return (ri << 16) | (gi << 8) | bi;
}

void release_image_buffers(std::vector<ImageBuffer>& v) {
    for (auto& b : v) { std::free(b.bits); b.bits = nullptr; b.width = 0; b.height = 0; }
    v.clear();
}

std::vector<ColorImage> convert_alpha_add_to_overlay(const std::vector<ColorImage>& layers) {
    const int L = static_cast<int>(layers.size());
    if (L == 0) return {};
    const int W = layers[0].width(), H = layers[0].height();
    std::vector<ColorImage> out(L, ColorImage(W, H));
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            for (int idx = 0; idx < L; ++idx) {
                double sum_current = 0.0;
                for (int i = 0; i <= idx; ++i) sum_current += layers[i].a().at(x, y);
                const double a_idx = layers[idx].a().at(x, y);
                const double oa = (sum_current < 1e-16) ? 1.0 : a_idx / sum_current;
                out[idx].r().at(x, y) = layers[idx].r().at(x, y);
                out[idx].g().at(x, y) = layers[idx].g().at(x, y);
                out[idx].b().at(x, y) = layers[idx].b().at(x, y);
                out[idx].a().at(x, y) = oa;
            }
        }
    }
    return out;
}

std::vector<ColorImage> matte_refinement(const ColorImage& original,
                                         const std::vector<ColorImage>& layers,
                                         const std::vector<ColorKernel>& kernels,
                                         const IBackend& backend) {
    const int L = static_cast<int>(layers.size());
    if (L == 0) return {};
    const int W = original.width(), H = original.height();
    const int P = W * H;
    const int radius = 60 * std::min(W, H) / 1000;

    std::vector<GrayImage> alpha_inputs(L);
    for (int i = 0; i < L; ++i) alpha_inputs[i] = layers[i].a();

    std::vector<GrayImage> refined_alphas;
    backend.guided_filter_batch(alpha_inputs, original, radius,
                                params::kGuidedFilterEps, refined_alphas);
    if ((int)refined_alphas.size() != L) {
        std::fprintf(stderr, "[unmix] matte_refinement: guided_filter_batch returned %zu layers, expected %d\n",
                     refined_alphas.size(), L);
        return {};
    }

    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
        double s = 0.0;
        for (int i = 0; i < L; ++i) {
            refined_alphas[i].at(x, y) = std::clamp(refined_alphas[i].at(x, y), 0.0, 1.0);
            s += refined_alphas[i].at(x, y);
        }
        if (s > 1e-16) for (int i = 0; i < L; ++i) refined_alphas[i].at(x, y) /= s;
    }

    BatchSolveRequest req;
    req.num_pixels     = P;
    req.num_layers     = L;
    req.kernels        = kernels;
    req.max_alm_iter   = params::kMaxALMIter;
    req.max_inner_iter = params::kMaxInnerEval;
    req.epsilon        = params::kALMEpsilon;
    req.target_colors.resize(static_cast<std::size_t>(P) * 3);
    req.target_alphas.resize(static_cast<std::size_t>(P) * L);
    req.initial_colors.resize(static_cast<std::size_t>(P) * L * 3);
    req.use_sparsity.assign(P, 0);
    req.use_target_alphas.assign(P, 1);

    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
        const int idx = y * W + x;
        req.target_colors[idx * 3 + 0] = original.r().at(x, y);
        req.target_colors[idx * 3 + 1] = original.g().at(x, y);
        req.target_colors[idx * 3 + 2] = original.b().at(x, y);
        for (int i = 0; i < L; ++i) {
            req.target_alphas[static_cast<std::size_t>(idx) * L + i] = refined_alphas[i].at(x, y);
            req.initial_colors[static_cast<std::size_t>(idx) * L * 3 + i * 3 + 0] = layers[i].r().at(x, y);
            req.initial_colors[static_cast<std::size_t>(idx) * L * 3 + i * 3 + 1] = layers[i].g().at(x, y);
            req.initial_colors[static_cast<std::size_t>(idx) * L * 3 + i * 3 + 2] = layers[i].b().at(x, y);
        }
    }

    const BatchSolveResult r = backend.solve_batch(req);
    if (r.solution.size() != static_cast<std::size_t>(P) * 4 * L) {
        std::fprintf(stderr, "[unmix] matte_refinement: solve_batch returned %zu, expected %zu\n",
                     r.solution.size(), static_cast<std::size_t>(P) * 4 * L);
        return {};
    }

    std::vector<ColorImage> refined(L, ColorImage(W, H));
    for (int idx = 0; idx < P; ++idx) {
        const int x = idx % W, y = idx / W;
        const double* sol = r.solution.data() + static_cast<std::size_t>(idx) * 4 * L;
        for (int i = 0; i < L; ++i) {
            refined[i].r().at(x, y) = sol[L + i * 3 + 0];
            refined[i].g().at(x, y) = sol[L + i * 3 + 1];
            refined[i].b().at(x, y) = sol[L + i * 3 + 2];
            refined[i].a().at(x, y) = sol[i];
        }
    }
    return refined;
}

ColorImage force_opaque(const ColorImage& img) {
    ColorImage out = img;
    std::fill(out.a().data(),
              out.a().data() + static_cast<std::size_t>(img.width()) * img.height(),
              1.0);
    return out;
}

}

void run(const std::uint8_t* bgra, int w, int h,
         UnmixedData& out,
         const IBackend& backend,
         bool verbose) {
    out.data_num = 0;
    out.data = nullptr;
    out.well_represented_map_num = 0;
    out.well_represented_map = nullptr;
    out.temporary_weight_map_num = 0;
    out.temporary_weight_map = nullptr;

    if (!bgra || w <= 0 || h <= 0) {
        std::fprintf(stderr, "[unmix] invalid input (bgra=%p, w=%d, h=%d)\n",
                     (const void*)bgra, w, h);
        return;
    }

    const ColorImage original = io::load_bgra(bgra, w, h);

    if (verbose) std::cout << "[unmix] luminance\n";
    const GrayImage gray     = backend.luminance(original);
    const GrayImage grad_mag = backend.gradient_magnitude(gray);

    std::vector<ColorKernel> kernels;
    std::vector<std::uint8_t> well_repr(static_cast<std::size_t>(w) * h, 0);
    std::vector<ImageBuffer> well_maps;
    std::vector<ImageBuffer> weight_maps;

    const double tau2 = params::kTau * params::kTau;

    int iteration = 0;
    while (true) {
        ++iteration;

        double bins[kNumBins][kNumBins][kNumBins] = {};

        std::vector<int> pending;
        pending.reserve(static_cast<std::size_t>(w) * h);
        for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
            const int idx = y * w + x;
            if (!well_repr[static_cast<std::size_t>(idx)]) pending.push_back(idx);
        }
        const int Np = static_cast<int>(pending.size());

        std::vector<double> scores(Np, DBL_MAX);
        int n_solver = 0;

        if (!kernels.empty() && Np > 0) {
            const int L = static_cast<int>(kernels.size());

            std::unordered_map<std::uint32_t, int> key_to_unique;
            key_to_unique.reserve(static_cast<std::size_t>(Np));
            std::vector<int> unique_pixels;
            std::vector<std::uint32_t> pending_keys(Np);

            for (int k = 0; k < Np; ++k) {
                const int idx = pending[k];
                const int x = idx % w, y = idx / w;
                const std::uint32_t key = quantize_rgb(
                    original.r().at(x, y),
                    original.g().at(x, y),
                    original.b().at(x, y));
                pending_keys[k] = key;
                auto it = key_to_unique.find(key);
                if (it == key_to_unique.end()) {
                    key_to_unique.emplace(key, static_cast<int>(unique_pixels.size()));
                    unique_pixels.push_back(idx);
                }
            }

            const int Nu = static_cast<int>(unique_pixels.size());

            std::vector<double> unique_energy(Nu, DBL_MAX);
            std::vector<int> solver_indices;
            solver_indices.reserve(Nu);

            for (int k = 0; k < Nu; ++k) {
                solver_indices.push_back(k);
            }

            const int Ns = static_cast<int>(solver_indices.size());
            n_solver = Ns;

            if (Ns > 0) {
                BatchSolveRequest req;
                req.num_pixels     = Ns;
                req.num_layers     = L;
                req.kernels        = kernels;
                req.max_alm_iter   = params::kMaxALMIter;
                req.max_inner_iter = params::kMaxInnerEval;
                req.epsilon        = params::kALMEpsilon;
                req.target_colors.resize(static_cast<std::size_t>(Ns) * 3);
                req.use_sparsity.assign(Ns, 0);
                req.use_target_alphas.assign(Ns, 0);

                for (int k = 0; k < Ns; ++k) {
                    const int idx = unique_pixels[solver_indices[k]];
                    const int x = idx % w, y = idx / w;
                    req.target_colors[k * 3 + 0] = original.r().at(x, y);
                    req.target_colors[k * 3 + 1] = original.g().at(x, y);
                    req.target_colors[k * 3 + 2] = original.b().at(x, y);
                }

                const BatchSolveResult solver_res = backend.solve_batch(req);
                if (solver_res.solution.size() == static_cast<std::size_t>(Ns) * 4 * L) {
                    for (int k = 0; k < Ns; ++k) {
                        const int up = solver_indices[k];
                        const double* sol = solver_res.solution.data() + static_cast<std::size_t>(k) * 4 * L;
                        const Eigen::Map<const VectorXd> sol_v(sol, 4 * L);
                        unique_energy[up] = energy_function(sol_v.head(L), sol_v.tail(3 * L), kernels, 0.0, false);
                    }
                } else {
                    std::fprintf(stderr, "[unmix] iter %d: solve_batch returned invalid solution\n", iteration);
                }
            }

            for (int k = 0; k < Np; ++k) {
                const int up = key_to_unique.at(pending_keys[k]);
                scores[k] = unique_energy[up];
            }

            if (verbose)
                std::cout << "[unmix] iter " << iteration
                          << " pending=" << Np
                          << " unique=" << Nu
                          << " solver=" << Ns << "\n";
        }

        for (int k = 0; k < Np; ++k) {
            const int idx = pending[k];
            const int x = idx % w, y = idx / w;
            const double score = scores[k];
            if (score < tau2) {
                well_repr[static_cast<std::size_t>(idx)] = 1;
                continue;
            }
            const Vector3d pixel(original.r().at(x, y), original.g().at(x, y), original.b().at(x, y));
            const Bin3 bin = get_bin(pixel);
            const double v = std::exp(-grad_mag.at(x, y)) * (1.0 - std::exp(-score));
            bins[bin.r][bin.g][bin.b] += v;
        }

        {
            GrayImage wm(w, h);
            for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x)
                wm.at(x, y) = well_repr[static_cast<std::size_t>(y) * w + x] ? 1.0 : 0.0;
            ImageBuffer ib;
            io::save_gray_bgra(wm, ib);
            well_maps.push_back(ib);
        }

        int cnt = 0;
        for (std::size_t i = 0; i < well_repr.size(); ++i) if (well_repr[i]) ++cnt;
        const double frac = static_cast<double>(cnt) / (w * h);
        if (verbose)
            std::cout << "[unmix] iter " << iteration << " well_repr " << cnt << "/" << w * h
                      << " (" << (frac * 100.0) << "%)\n";
        if (frac > params::kAlmostThreshold) break;

        int mr = 0, mg = 0, mb = 0;
        double max_vote = -1.0;
        for (int r = 0; r < kNumBins; ++r)
            for (int g = 0; g < kNumBins; ++g)
                for (int b = 0; b < kNumBins; ++b)
                    if (bins[r][g][b] >= max_vote) {
                        max_vote = bins[r][g][b];
                        mr = r; mg = g; mb = b;
                    }

        int seed_x = -1, seed_y = -1;
        double max_score = 0.0;
        for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
            if (well_repr[static_cast<std::size_t>(y) * w + x]) continue;
            const Vector3d pixel(original.r().at(x, y), original.g().at(x, y), original.b().at(x, y));
            const Bin3 bin = get_bin(pixel);
            if (bin.r != mr || bin.g != mg || bin.b != mb) continue;
            int nc = 0;
            for (int oy = -params::kNeighborhoodRadius; oy <= params::kNeighborhoodRadius; ++oy) {
                if (y + oy < 0 || y + oy >= h) continue;
                for (int ox = -params::kNeighborhoodRadius; ox <= params::kNeighborhoodRadius; ++ox) {
                    if (x + ox < 0 || x + ox >= w) continue;
                    if (well_repr[static_cast<std::size_t>(y + oy) * w + (x + ox)]) continue;
                    const Vector3d c(
                        original.r().at(x + ox, y + oy),
                        original.g().at(x + ox, y + oy),
                        original.b().at(x + ox, y + oy));
                    const Bin3 nb = get_bin(c);
                    if (nb.r == mr && nb.g == mg && nb.b == mb) ++nc;
                }
            }
            if (nc == 0) continue;
            const double s = static_cast<double>(nc) * std::exp(-grad_mag.at(x, y));
            if (s > max_score) { max_score = s; seed_x = x; seed_y = y; }
        }
        if (seed_x < 0 || seed_y < 0) {
            if (verbose) std::cout << "[unmix] no seed found, terminating early\n";
            break;
        }

        if (verbose) std::cout << "[unmix] seed (" << seed_x << "," << seed_y << ")\n";
        const GrayImage weight_map = backend.guided_filter_kernel(
            gray, seed_x, seed_y, params::kNeighborhoodRadius, params::kGuidedFilterKernelEps, true);
        {
            GrayImage wm = weight_map;
            host::scale_to_unit(wm);
            ImageBuffer ib;
            io::save_gray_bgra(wm, ib);
            weight_maps.push_back(ib);
        }

        Vector3d mu; Matrix3d sigma;
        backend.compute_normal_distribution(original, weight_map, mu, sigma);

        ColorKernel k;
        k.mu        = mu;
        k.sigma_inv = sigma.inverse();
        k.seed_x    = seed_x;
        k.seed_y    = seed_y;
        kernels.push_back(k);
        if (verbose)
            std::cout << "[unmix] kernel " << kernels.size() << " mu " << mu.transpose() << "\n";
    }

    const int L = static_cast<int>(kernels.size());
    const int P = w * h;
    if (verbose) std::cout << "[unmix] final layers " << L << "\n";

    if (L == 0) {
        if (verbose) std::cout << "[unmix] no kernels found, returning empty output\n";
        release_image_buffers(well_maps);
        release_image_buffers(weight_maps);
        return;
    }

    BatchSolveRequest req;
    req.num_pixels     = P;
    req.num_layers     = L;
    req.kernels        = kernels;
    req.max_alm_iter   = params::kMaxALMIter;
    req.max_inner_iter = params::kMaxInnerEval;
    req.epsilon        = params::kALMEpsilon;
    req.target_colors.resize(static_cast<std::size_t>(P) * 3);
    req.use_sparsity.assign(P, 0);
    req.use_target_alphas.assign(P, 0);

    for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
        const int idx = y * w + x;
        req.target_colors[idx * 3 + 0] = original.r().at(x, y);
        req.target_colors[idx * 3 + 1] = original.g().at(x, y);
        req.target_colors[idx * 3 + 2] = original.b().at(x, y);
    }

    const BatchSolveResult final_res = backend.solve_batch(req);
    if (final_res.solution.size() != static_cast<std::size_t>(P) * 4 * L) {
        std::fprintf(stderr, "[unmix] final solve_batch returned %zu, expected %zu\n",
                     final_res.solution.size(), static_cast<std::size_t>(P) * 4 * L);
        release_image_buffers(well_maps);
        release_image_buffers(weight_maps);
        return;
    }

    std::vector<ColorImage> layers(L, ColorImage(w, h));
    for (int idx = 0; idx < P; ++idx) {
        const int x = idx % w, y = idx / w;
        const double* sol = final_res.solution.data() + static_cast<std::size_t>(idx) * 4 * L;
        for (int i = 0; i < L; ++i) {
            layers[i].r().at(x, y) = sol[L + i * 3 + 0];
            layers[i].g().at(x, y) = sol[L + i * 3 + 1];
            layers[i].b().at(x, y) = sol[L + i * 3 + 2];
            layers[i].a().at(x, y) = sol[i];
        }
    }

    if (verbose) std::cout << "[unmix] overlay + refinement\n";
    const auto overlay_layers  = convert_alpha_add_to_overlay(layers);
    const auto refined         = matte_refinement(original, layers, kernels, backend);
    const auto refined_overlay = convert_alpha_add_to_overlay(refined);

    if ((int)overlay_layers.size() != L || (int)refined.size() != L || (int)refined_overlay.size() != L) {
        std::fprintf(stderr, "[unmix] post-processing produced inconsistent layer count\n");
        release_image_buffers(well_maps);
        release_image_buffers(weight_maps);
        return;
    }

    out.data_num = L;
    out.data     = new UnmixedLayer[L];
    for (int i = 0; i < L; ++i) {
        io::save_color_bgra(force_opaque(layers[i]),  out.data[i].output);
        io::save_gray_bgra (layers[i].a(),            out.data[i].output_a);
        io::save_color_bgra(overlay_layers[i],        out.data[i].overlay);
        io::save_color_bgra(force_opaque(refined[i]), out.data[i].refined);
        io::save_gray_bgra (refined[i].a(),           out.data[i].refined_a);
        io::save_color_bgra(refined_overlay[i],       out.data[i].refined_overlay);
    }

    out.well_represented_map_num = static_cast<int>(well_maps.size());
    out.well_represented_map = well_maps.empty() ? nullptr : new ImageBuffer[well_maps.size()];
    for (std::size_t i = 0; i < well_maps.size(); ++i) out.well_represented_map[i] = well_maps[i];

    out.temporary_weight_map_num = static_cast<int>(weight_maps.size());
    out.temporary_weight_map = weight_maps.empty() ? nullptr : new ImageBuffer[weight_maps.size()];
    for (std::size_t i = 0; i < weight_maps.size(); ++i) out.temporary_weight_map[i] = weight_maps[i];
}

void run(const std::uint8_t* bgra, int w, int h,
         UnmixedData& out,
         bool verbose) {
    run(bgra, w, h, out, cpu_backend(), verbose);
}

void free_unmixed_data(UnmixedData& data) {
    if (data.data) {
        for (int i = 0; i < data.data_num; ++i) {
            auto& d = data.data[i];
            std::free(d.output.bits);           d.output.bits           = nullptr;
            std::free(d.output_a.bits);         d.output_a.bits         = nullptr;
            std::free(d.overlay.bits);          d.overlay.bits          = nullptr;
            std::free(d.refined.bits);          d.refined.bits          = nullptr;
            std::free(d.refined_a.bits);        d.refined_a.bits        = nullptr;
            std::free(d.refined_overlay.bits);  d.refined_overlay.bits  = nullptr;
        }
        delete[] data.data;
        data.data = nullptr;
    }
    if (data.well_represented_map) {
        for (int i = 0; i < data.well_represented_map_num; ++i)
            std::free(data.well_represented_map[i].bits);
        delete[] data.well_represented_map;
        data.well_represented_map = nullptr;
    }
    if (data.temporary_weight_map) {
        for (int i = 0; i < data.temporary_weight_map_num; ++i)
            std::free(data.temporary_weight_map[i].bits);
        delete[] data.temporary_weight_map;
        data.temporary_weight_map = nullptr;
    }
    data.data_num = 0;
    data.well_represented_map_num = 0;
    data.temporary_weight_map_num = 0;
}

}