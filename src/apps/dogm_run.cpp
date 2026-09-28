// dogm_run.cpp - standalone DOGM velocity estimator.
//
// Reads VoxelNeXt detection JSONL plus the pose cache, runs the corrected
// particle DOGM (src/dogm_rfs.hpp) and writes per-scene JSONL in exactly the
// same schema as the input, with each box's `velocity` replaced by the DOGM
// estimate (global frame). That keeps it directly comparable with the detector
// output and with the archived legacy DOGM results under tools/eval_velocity.py.
//
// No tracker is involved here on purpose: this binary measures DOGM in
// isolation, which is the piece the project never had a measurement for.
//
// Build: see CMakeLists.txt (or)
//   g++ -std=c++17 -O2 -D_USE_MATH_DEFINES src/apps/dogm_run.cpp \
//       -o build/dogm_run.exe -Isrc -I/c/msys64/ucrt64/include/eigen3

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include "dogm_rfs.hpp"
#include "json.hpp"

using json = nlohmann::json;
namespace fs = std::filesystem;

static const char* ALGO_VERSION = "dgmtrack-dogm-2.0.0-rfs";

// ------------------------------------------------------------------ helpers
static Eigen::Quaterniond quat_wxyz(const std::vector<double>& q) {
    return Eigen::Quaterniond(q[0], q[1], q[2], q[3]).normalized();
}
static double yaw_of(const Eigen::Quaterniond& q) {
    const Eigen::Matrix3d R = q.toRotationMatrix();
    return std::atan2(R(1, 0), R(0, 0));
}

struct PoseEntry {
    int64_t ts = 0;
    std::string file, scene;
    Eigen::Vector3d cs_t, ep_t;
    Eigen::Quaterniond cs_q, ep_q;
};

static std::vector<Eigen::Vector2f> load_lidar_xy(const std::string& path) {
    std::vector<Eigen::Vector2f> pts;
    std::ifstream f(path, std::ios::binary);
    if (!f) return pts;
    f.seekg(0, std::ios::end);
    const size_t bytes = (size_t)f.tellg();
    f.seekg(0, std::ios::beg);
    if (bytes % 4) return pts;
    std::vector<float> buf(bytes / 4);
    f.read((char*)buf.data(), bytes);
    const size_t nf = buf.size();
    const int stride = (nf % 5 == 0) ? 5 : ((nf % 4 == 0) ? 4 : 6);
    pts.reserve(nf / stride);
    for (size_t i = 0; i + stride <= nf; i += stride)
        pts.emplace_back(buf[i], buf[i + 1]);
    return pts;
}

struct DetBox {
    std::vector<double> t{0, 0, 0}, s{1, 1, 1}, q{1, 0, 0, 0};
    std::string name, attr;
    double score = 1.0;
    bool has_vel = false;
    double vx = 0, vy = 0;
};
struct Frame {
    std::string sample_token, sd_token;
    bool is_key = false;
    int64_t ts = -1;
    std::vector<DetBox> boxes;
};

static const std::vector<std::string> TRACKING_CLASSES = {
    "car", "truck", "bus", "trailer", "pedestrian", "motorcycle", "bicycle"};
static bool is_tracking_class(const std::string& s) {
    for (const auto& c : TRACKING_CLASSES)
        if (c == s) return true;
    return false;
}

static std::vector<Frame> load_scene(const std::string& path) {
    std::vector<Frame> out;
    std::ifstream f(path);
    if (!f) return out;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        json j;
        try {
            j = json::parse(line);
        } catch (...) { continue; }
        Frame fr;
        fr.sample_token = j.value("sample_token", std::string());
        fr.sd_token = j.value("sd_token", std::string());
        fr.is_key = j.value("is_key_frame", false);
        fr.ts = j.value("timestamp", (int64_t)-1);
        if (j.contains("boxes"))
            for (const auto& b : j["boxes"]) {
                DetBox d;
                if (b.contains("translation")) d.t = b["translation"].get<std::vector<double>>();
                if (b.contains("size")) d.s = b["size"].get<std::vector<double>>();
                if (b.contains("rotation")) d.q = b["rotation"].get<std::vector<double>>();
                d.name = b.value("detection_name", std::string());
                d.attr = b.value("attribute_name", std::string());
                d.score = b.value("detection_score", 1.0);
                if (b.contains("velocity") && b["velocity"].is_array() &&
                    b["velocity"].size() >= 2) {
                    d.vx = b["velocity"][0].get<double>();
                    d.vy = b["velocity"][1].get<double>();
                    d.has_vel = std::isfinite(d.vx) && std::isfinite(d.vy);
                }
                fr.boxes.push_back(std::move(d));
            }
        out.push_back(std::move(fr));
    }
    std::sort(out.begin(), out.end(),
              [](const Frame& a, const Frame& b) { return a.ts < b.ts; });
    return out;
}

// ---------------------------------------------------------------------- main
int main(int argc, char** argv) {
    std::string pose_cache, det_dir, out_dir, dataroot, scenes_arg;
    double score_thr = 0.0;
    dogm::Config cfg;
    bool write_all_frames = false;
    bool use_det_init = true;   // seed births from the detector velocity [3b]

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() { return (i + 1 < argc) ? std::string(argv[++i]) : std::string(); };
        if (a == "--pose_cache") pose_cache = val();
        else if (a == "--det_dir") det_dir = val();
        else if (a == "--out_dir") out_dir = val();
        else if (a == "--dataroot") dataroot = val();
        else if (a == "--scenes") scenes_arg = val();
        else if (a == "--score_thr") score_thr = std::stod(val());
        else if (a == "--particles_per_box") cfg.particles_per_box = std::stoi(val());
        else if (a == "--sigma_v_proc") cfg.sigma_v_proc = std::stod(val());
        else if (a == "--sigma_p_proc") cfg.sigma_p_proc = std::stod(val());
        else if (a == "--occ_measured") cfg.occ_measured = std::stod(val());
        else if (a == "--occ_unmeasured") cfg.occ_unmeasured = std::stod(val());
        else if (a == "--birth_mass_share") cfg.birth_mass_share = std::stod(val());
        else if (a == "--resample_neff_ratio") cfg.resample_neff_ratio = std::stod(val());
        else if (a == "--birth_vel_sigma") cfg.birth_vel_sigma = std::stod(val());
        else if (a == "--birth_seed_fraction") cfg.birth_seed_fraction = std::stod(val());
        else if (a == "--birth_static_fraction") cfg.birth_static_fraction = std::stod(val());
        else if (a == "--meas_dilate") cfg.meas_dilate = std::stoi(val());
        else if (a == "--vel_std_abs") cfg.vel_std_abs = std::stod(val());
        else if (a == "--vel_std_rel") cfg.vel_std_rel = std::stod(val());
        else if (a == "--prune_pad_m") cfg.prune_pad_m = std::stod(val());
        else if (a == "--resolution") cfg.grid.resolution = std::stod(val());
        else if (a == "--seed") cfg.seed = (uint64_t)std::stoull(val());
        else if (a == "--no_resample") cfg.resample = false;
        else if (a == "--write_all") write_all_frames = true;
        else if (a == "--no_det_init") use_det_init = false;
        else if (a == "--birth_init_fraction") cfg.birth_init_fraction = std::stod(val());
        else if (a == "--birth_init_sigma") cfg.birth_init_sigma = std::stod(val());
        else if (a == "--static_speed_thresh") cfg.static_speed_thresh = std::stod(val());
        else if (a == "--help") {
            std::cerr << "usage: dogm_run --pose_cache <json> --det_dir <dir> "
                         "--out_dir <dir> --dataroot <nuscenes root> [options]\n";
            return 0;
        }
    }
    if (pose_cache.empty() || det_dir.empty() || out_dir.empty() || dataroot.empty()) {
        std::cerr << "ERROR: --pose_cache, --det_dir, --out_dir and --dataroot "
                     "are required (see --help)\n";
        return 1;
    }
    cfg.grid.finalize();

    // ---- pose cache
    auto t_load = std::chrono::steady_clock::now();
    std::unordered_map<std::string, PoseEntry> poses;
    {
        std::ifstream f(pose_cache);
        if (!f) { std::cerr << "cannot open pose cache " << pose_cache << "\n"; return 1; }
        json j;
        f >> j;
        for (auto it = j["frames"].begin(); it != j["frames"].end(); ++it) {
            const auto& v = it.value();
            PoseEntry e;
            e.ts = v["ts"].get<int64_t>();
            e.file = v["file"].get<std::string>();
            e.scene = v["scene"].get<std::string>();
            auto ct = v["cs_t"].get<std::vector<double>>();
            auto et = v["ep_t"].get<std::vector<double>>();
            e.cs_t = Eigen::Vector3d(ct[0], ct[1], ct[2]);
            e.ep_t = Eigen::Vector3d(et[0], et[1], et[2]);
            e.cs_q = quat_wxyz(v["cs_q"].get<std::vector<double>>());
            e.ep_q = quat_wxyz(v["ep_q"].get<std::vector<double>>());
            poses.emplace(it.key(), std::move(e));
        }
    }
    std::cerr << "[dogm_run] pose cache: " << poses.size() << " frames ("
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load).count()
              << "s)\n";

    // ---- scene list
    std::vector<std::string> scene_files;
    if (!scenes_arg.empty()) {
        std::stringstream ss(scenes_arg);
        std::string s;
        while (std::getline(ss, s, ','))
            if (!s.empty()) scene_files.push_back(det_dir + "/" + s + ".jsonl");
    } else {
        for (const auto& e : fs::directory_iterator(det_dir))
            if (e.is_regular_file() && e.path().extension() == ".jsonl")
                scene_files.push_back(e.path().string());
        std::sort(scene_files.begin(), scene_files.end());
    }
    fs::create_directories(out_dir);
    std::cerr << "[dogm_run] " << scene_files.size() << " scenes\n";

    dogm::DogmRfs filter(cfg);
    size_t total_frames = 0, total_boxes = 0;
    const auto t_start = std::chrono::steady_clock::now();

    for (const auto& sf : scene_files) {
        const std::string scene = fs::path(sf).stem().string();
        auto frames = load_scene(sf);
        if (frames.empty()) { std::cerr << "  [skip] " << scene << "\n"; continue; }

        filter.reset();  // no state leaks across scenes (legacy batch mode did)
        bool have_prev = false;
        Eigen::Matrix3d R_prev = Eigen::Matrix3d::Identity();
        Eigen::Vector3d t_prev = Eigen::Vector3d::Zero();
        int64_t prev_ts = -1;

        std::ofstream out(out_dir + "/" + scene + ".jsonl", std::ios::trunc);

        for (const auto& fr : frames) {
            auto pit = poses.find(fr.sd_token);
            if (pit == poses.end()) continue;
            const PoseEntry& pe = pit->second;

            const Eigen::Matrix3d R_e = pe.ep_q.toRotationMatrix();
            const Eigen::Matrix3d R_cs = pe.cs_q.toRotationMatrix();
            const Eigen::Matrix3d R_cur = R_e * R_cs;
            const Eigen::Vector3d t_cur = pe.ep_t + R_e * pe.cs_t;

            Eigen::Matrix2d R_rel = Eigen::Matrix2d::Identity();
            Eigen::Vector2d t_rel = Eigen::Vector2d::Zero();
            if (have_prev) {
                const Eigen::Matrix3d R3 = R_cur.transpose() * R_prev;
                const Eigen::Vector3d t3 = R_cur.transpose() * (t_prev - t_cur);
                R_rel << R3(0, 0), R3(0, 1), R3(1, 0), R3(1, 1);
                t_rel << t3.x(), t3.y();
            }
            double dt = 0.05;
            if (prev_ts > 0) dt = std::clamp((fr.ts - prev_ts) / 1e6, 0.01, 0.5);

            // boxes: global -> LiDAR
            std::vector<dogm::BoxLidar2D> boxes;
            std::vector<Eigen::Vector2d> init_vel;  // [3b] initial velocity estimate
            std::vector<int> orig;
            boxes.reserve(fr.boxes.size());
            for (size_t i = 0; i < fr.boxes.size(); ++i) {
                const DetBox& d = fr.boxes[i];
                if (!is_tracking_class(d.name) || d.score < score_thr) continue;
                const Eigen::Vector3d pg(d.t[0], d.t[1], d.t[2]);
                const Eigen::Quaterniond qg = quat_wxyz(d.q);
                const Eigen::Vector3d p_ego = pe.ep_q.inverse() * (pg - pe.ep_t);
                const Eigen::Vector3d p_l = pe.cs_q.inverse() * (p_ego - pe.cs_t);
                const Eigen::Quaterniond q_l = pe.cs_q.inverse() * (pe.ep_q.inverse() * qg);
                if (p_l.x() * p_l.x() + p_l.y() * p_l.y() <= 4.0) continue;  // ego self-return
                dogm::BoxLidar2D b;
                b.cx = p_l.x(); b.cy = p_l.y();
                b.w = d.s[0]; b.l = d.s[1];
                b.yaw = yaw_of(q_l);
                boxes.push_back(b);
                orig.push_back((int)i);
                // The detector's own velocity head is the initial estimate the
                // birth mixture is built around; without one the box falls back
                // to the wide prior.
                Eigen::Vector2d v0 = Eigen::Vector2d::Zero();
                if (use_det_init && d.has_vel) {
                    const Eigen::Vector3d vg(d.vx, d.vy, 0.0);
                    const Eigen::Vector3d vl =
                        pe.cs_q.inverse() * (pe.ep_q.inverse() * vg);
                    v0 << vl.x(), vl.y();
                }
                init_vel.push_back(v0);
            }

            const auto pts = load_lidar_xy(dataroot + "/" + pe.file);
            std::vector<dogm::BoxVelocity> vel;
            // an empty init vector means "no initial estimate" - the birth
            // model then falls back to the wide prior for every box
            filter.step(R_rel, t_rel, dt, pts, boxes,
                        use_det_init ? init_vel : std::vector<Eigen::Vector2d>{}, vel);

            have_prev = true;
            R_prev = R_cur;
            t_prev = t_cur;
            prev_ts = fr.ts;
            ++total_frames;

            if (!fr.is_key && !write_all_frames) continue;

            // velocities back to the global frame, written in the input schema
            std::vector<std::array<double, 2>> vg(fr.boxes.size(), {0.0, 0.0});
            std::vector<char> vvalid(fr.boxes.size(), 0);
            std::vector<double> vconf(fr.boxes.size(), 0.0);
            for (size_t k = 0; k < boxes.size(); ++k) {
                if (!vel[k].valid) continue;
                const Eigen::Vector3d vl(vel[k].v.x(), vel[k].v.y(), 0.0);
                const Eigen::Vector3d g = pe.ep_q * (pe.cs_q * vl);
                vg[orig[k]] = {g.x(), g.y()};
                vvalid[orig[k]] = 1;
                vconf[orig[k]] = vel[k].conf;
                ++total_boxes;
            }

            json line;
            line["sample_token"] = fr.sample_token;
            line["sd_token"] = fr.sd_token;
            line["is_key_frame"] = fr.is_key;
            line["timestamp"] = fr.ts;
            json arr = json::array();
            for (size_t i = 0; i < fr.boxes.size(); ++i) {
                const DetBox& d = fr.boxes[i];
                arr.push_back({{"translation", d.t},
                               {"size", d.s},
                               {"rotation", d.q},
                               {"velocity", {vg[i][0], vg[i][1]}},
                               {"detection_name", d.name},
                               {"detection_score", d.score},
                               {"attribute_name", d.attr},
                               {"dogm_valid", (bool)vvalid[i]},
                               {"dogm_conf", vconf[i]}});
            }
            line["boxes"] = arr;
            out << line.dump() << "\n";
        }
        std::cerr << "  " << scene << " (" << frames.size() << " frames, "
                  << filter.particle_count() << " particles)\n";
    }

    const double secs =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    std::cerr << "[dogm_run] " << total_frames << " frames in " << secs << "s ("
              << (total_frames ? secs / total_frames * 1000.0 : 0.0) << " ms/frame), "
              << total_boxes << " box velocities\n";

    // run metadata, so a result folder always says which algorithm produced it
    json meta = {
        {"algo_version", ALGO_VERSION},
        {"seed", cfg.seed},
        {"score_thr", score_thr},
        {"dogm", {{"particles_per_box", cfg.particles_per_box},
                  {"max_particles", cfg.max_particles},
                  {"sigma_v_proc", cfg.sigma_v_proc},
                  {"sigma_p_proc", cfg.sigma_p_proc},
                  {"occ_measured", cfg.occ_measured},
                  {"occ_unmeasured", cfg.occ_unmeasured},
                  {"meas_dilate", cfg.meas_dilate},
                  {"resample", cfg.resample},
                  {"resample_neff_ratio", cfg.resample_neff_ratio},
                  {"birth_mass_share", cfg.birth_mass_share},
                  {"birth_vel_sigma", cfg.birth_vel_sigma},
                  {"birth_seed_fraction", cfg.birth_seed_fraction},
                  {"birth_static_fraction", cfg.birth_static_fraction},
                  {"birth_init_fraction", cfg.birth_init_fraction},
                  {"birth_init_sigma", cfg.birth_init_sigma},
                  {"static_speed_thresh", cfg.static_speed_thresh},
                  {"use_det_init", use_det_init},
                  {"sigma_v_floor", cfg.sigma_v_floor},
                  {"vel_std_abs", cfg.vel_std_abs},
                  {"vel_std_rel", cfg.vel_std_rel},
                  {"prune_pad_m", cfg.prune_pad_m},
                  {"resolution", cfg.grid.resolution}}},
        {"runtime_s", secs},
        {"frames", total_frames},
        {"det_dir", det_dir},
        {"pose_cache", pose_cache}};
    std::ofstream mf(out_dir + "/run_meta.json");
    mf << meta.dump(2) << "\n";
    return 0;
}
