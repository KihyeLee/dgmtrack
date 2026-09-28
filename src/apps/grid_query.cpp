// Causal tracker-independent readout of the recovered particle DOGM.
// Plans contain only current submitted boxes and past-only extrapolations.
// The particle birth model never receives the track velocity used for scoring.
#include <chrono>
#include <filesystem>
#include <iostream>
#include <map>
#include "dogm_rfs.hpp"
#include "common/nusc_io.hpp"

using nusc::json;

int main(int argc, char** argv) try {
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) throw std::runtime_error("missing argument value");
        args[argv[i]] = argv[i + 1];
    }
    for (const auto& k : {"--poses", "--plan", "--dataroot", "--out"})
        if (!args.count(k)) throw std::runtime_error(std::string("required: ") + k);
    std::ifstream pf(args.at("--plan"));
    json plan; pf >> plan;
    const std::string scene = plan.at("scene");
    const auto& frames = plan.at("frames");
    if (frames.empty()) throw std::runtime_error("empty scene plan");
    std::unordered_map<std::string, nusc::PoseEntry> poses;
    if (!nusc::load_pose_cache(args.at("--poses"), poses))
        throw std::runtime_error("cannot load pose cache");
    std::map<std::string, json> keys;
    for (const auto& f : frames) keys[f.at("sd_token")] = f;
    const int64_t begin = frames.front().at("timestamp_us");
    const int64_t end = frames.back().at("timestamp_us");
    std::vector<std::pair<std::string, nusc::PoseEntry>> timeline;
    for (const auto& kv : poses)
        if (kv.second.scene == scene && kv.second.ts >= begin && kv.second.ts <= end)
            timeline.push_back(kv);
    std::sort(timeline.begin(), timeline.end(), [](const auto& a, const auto& b) {
        return a.second.ts < b.second.ts;
    });
    dogm::Config cfg;
    cfg.seed = 12345;
    cfg.birth_seed_fraction = 0.0;
    cfg.birth_init_fraction = 0.0;
    dogm::DogmRfs filter(cfg);
    json output = {{"scene", scene}, {"samples", json::object()}};
    json last_targets = json::array();
    int64_t last_key = begin, prev_ts = 0;
    Eigen::Matrix3d prev_R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d prev_t = Eigen::Vector3d::Zero();
    const auto started = std::chrono::steady_clock::now();
    size_t num_sweeps = 0, num_points = 0;
    for (const auto& entry : timeline) {
        const auto& pe = entry.second;
        const auto ki = keys.find(entry.first);
        const bool is_key = ki != keys.end();
        if (is_key) {
            last_targets = ki->second.at("targets");
            last_key = pe.ts;
        }
        Eigen::Matrix3d R; Eigen::Vector3d t;
        nusc::lidar_pose(pe, R, t);
        Eigen::Matrix2d rel_R = Eigen::Matrix2d::Identity();
        Eigen::Vector2d rel_t = Eigen::Vector2d::Zero();
        if (prev_ts) nusc::relative_2d(R, t, prev_R, prev_t, rel_R, rel_t);
        const double dt = prev_ts ? (pe.ts - prev_ts) * 1e-6 : 0.05;
        if (!(dt > 0 && dt <= 1.0)) throw std::runtime_error("invalid frame interval");
        std::vector<dogm::BoxLidar2D> boxes;
        for (const auto& target : last_targets) {
            const auto& b = target.at("box");
            nusc::DetBox d;
            d.t = b.at("translation").get<std::vector<double>>();
            d.s = b.at("size").get<std::vector<double>>();
            d.q = b.at("rotation").get<std::vector<double>>();
            d.name = b.at("tracking_name");
            const double age = (pe.ts - last_key) * 1e-6;
            d.t[0] += target.at("velocity")[0].get<double>() * age;
            d.t[1] += target.at("velocity")[1].get<double>() * age;
            const auto local = nusc::box_to_lidar(d, pe, 0);
            boxes.push_back({local.cx, local.cy, local.w, local.l, local.yaw});
        }
        const auto path = std::filesystem::path(args.at("--dataroot")) / pe.file;
        if (!std::filesystem::is_regular_file(path) || std::filesystem::file_size(path) % 20)
            throw std::runtime_error("missing or malformed LiDAR: " + path.string());
        const auto points = nusc::load_lidar_xy(path.string());
        if (points.empty()) throw std::runtime_error("empty LiDAR: " + path.string());
        std::vector<dogm::BoxVelocity> estimates;
        filter.step(rel_R, rel_t, dt, points, boxes, {}, estimates);
        if (is_key) {
            json values = json::array();
            for (size_t i = 0; i < boxes.size(); ++i) {
                const auto& e = estimates.at(i);
                const Eigen::Vector3d vg = R * Eigen::Vector3d(e.v.x(), e.v.y(), 0);
                values.push_back({{"tracking_id", last_targets[i]["box"]["tracking_id"]},
                    {"candidate", last_targets[i]["candidate"]},
                    {"support", filter.box_support(boxes[i])},
                    {"valid", e.valid}, {"particles", e.n_particles},
                    {"velocity", {vg.x(), vg.y()}}, {"confidence", e.conf}});
            }
            output["samples"][ki->second.at("sample_token").get<std::string>()] = values;
        }
        prev_R = R; prev_t = t; prev_ts = pe.ts;
        ++num_sweeps; num_points += points.size();
    }
    if (output["samples"].size() != frames.size())
        throw std::runtime_error("not all planned keyframes were processed");
    output["runtime_s"] = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    output["sweeps"] = num_sweeps;
    output["points"] = num_points;
    output["seed"] = cfg.seed;
    output["birth_seed_fraction"] = cfg.birth_seed_fraction;
    output["birth_init_fraction"] = cfg.birth_init_fraction;
    const std::string temporary = args.at("--out") + ".tmp";
    { std::ofstream f(temporary); f << output.dump(); if (!f) throw std::runtime_error("write failed"); }
    std::filesystem::rename(temporary, args.at("--out"));
    std::cout << scene << " " << output["runtime_s"] << " s, " << num_sweeps << " sweeps\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "grid_query: " << e.what() << "\n";
    return 1;
}
