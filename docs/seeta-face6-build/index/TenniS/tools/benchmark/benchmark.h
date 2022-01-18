//
// Created by yang on 2019/10/9.
//

#ifndef TENSORSTACK_BENCH_H
#define TENSORSTACK_BENCH_H

#include "core/device.h"
#include "module/module.h"
#include "board/hook.h"
#include "runtime/workbench.h"
#include <string>
#include <chrono>
#include <algorithm>

namespace ts{

    struct Option{
        int loop_counts;
        int num_threads;
        DeviceType device;
        int id;
        std::string compile_option;
        int power_mode;
    };

    class DataBase {
    public:
        using self = DataBase;

        struct DataS {
            DataS() {
                op = "";
                name = "";
                pass_time = 0;
            }
            DataS(std::string op,std::string name,float pass_time)
                    :op(op), name(name), pass_time(pass_time){

            }

            std::string op;
            std::string name;
            float pass_time;
        };

        struct OpDataS {
            int count;
            float avg_time;
        };

        explicit DataBase(int count_num)
            : m_index(0)
            , m_count_num(count_num)
            , m_avg_pass_time(0){
            m_times.resize(count_num);
        }

        DataBase(const DataBase& other)
            : m_count_num(other.m_count_num)
            , m_index(other.m_index)
            , m_times(other.m_times)
            , time_name(other.time_name)
            , time_op(other.time_op)
            , time_op_vec(other.time_op_vec)
            , m_avg_pass_time(other.m_avg_pass_time){

        }

        DataBase &operator=(const DataBase& other){
            m_count_num = other.m_count_num;
            m_index = other.m_index;
            m_times = other.m_times;
            time_name = other.time_name;
            time_op = other.time_op;
            time_op_vec = other.time_op_vec;
            m_avg_pass_time = other.m_avg_pass_time;
            return *this;
        }

        ~DataBase(){
            clear();
        };

        void index_inch() {
            m_index++;
        }

        void add(const DataS& data) {
            m_times[m_index].emplace_back(data);
        }

        void clear(){
            m_index = 0;
            m_times.clear();
            time_name.clear();
            time_op.clear();
            time_op_vec.clear();
            m_avg_pass_time = 0.f;
        }

        void analyze() {
            if (m_times.empty())
                return;
            std::vector<std::string> names(m_times[0].size());
            std::vector<std::string> ops(m_times[0].size());
            std::vector<float> time_vec(m_times[0].size(), 0);
            std::vector<float> avg_time_vec(m_times[0].size(), 0);

            for (int i = 1; i < m_times.size(); i++){
                auto data_vec = m_times[i];
                for (int j = 0; j < data_vec.size(); j++){
                    time_vec[j] += data_vec[j].pass_time;
                    names[j] = data_vec[j].name;
                    ops[j] = data_vec[j].op;
                }
            }
            for (int i = 0; i < time_vec.size(); i++){
                auto avg_time = time_vec[i] / (m_count_num - 1);
                m_avg_pass_time += avg_time;
                avg_time_vec[i] = avg_time;
                auto name = names[i];
                auto op = ops[i];
                time_name[name] = avg_time;
                if (time_op.find(op) != time_op.end()) {
                    time_op[op].count++;
                    time_op[op].avg_time += avg_time;
                }
                else {
                    time_op[op].count = 1;
                    time_op[op].avg_time = avg_time;
                }
            }
            std::vector<std::pair<std::string, OpDataS>> vt(time_op.begin(), time_op.end());
            std::sort(vt.begin(), vt.end(), [](const std::pair<std::string, OpDataS>& a, const std::pair<std::string, OpDataS>& b) {return a.second.avg_time >= b.second.avg_time; });
            time_op_vec = vt;
        }

        void log(std::ostream &out) {
            int layer_num = time_name.size();
            out << "[loop_count : " << m_count_num << " ][layer_num : " << layer_num <<  " ]" << std::endl;
            int index = 0;
            auto data = m_times[0];
            out << "================layer info================" << std::endl;
            out << "[all avg pass time : " << m_avg_pass_time << " ]" << std::endl;
            for (int i = 0; i < layer_num; i++){
                auto op = data[i].op;
                auto name = data[i].name;
                float avg_time = time_name[name];
                out << "[index:" << index << ",op:" << op << ",name:" << name << " ][avg time : " << avg_time << " ]" << std::endl;
                index++;
            }
            out << "================ops info==================" << std::endl;
            for (int i = 0; i < time_op_vec.size(); i++){
                auto pair = time_op_vec[i];
                auto op = pair.first;
                int count = pair.second.count;
                auto avg_time = pair.second.avg_time;
                out << "[op:" << op << ",count:"  << count << "][avg time : " << avg_time << "]" << std::endl;
            }
            out << std::endl;
        }

    private:
        int m_count_num;
        int m_index;
        std::vector<std::vector<DataS>> m_times;
        std::map<std::string , float> time_name;
        std::map<std::string, OpDataS> time_op;
        std::vector<std::pair<std::string, OpDataS>> time_op_vec;
        float m_avg_pass_time;
    };

    class BenchMark{
    public:
        using microseconds = std::chrono::microseconds;
        using system_clock = std::chrono::system_clock;
        using time_point = decltype(system_clock::now());

    public:
        BenchMark(const Option& op, bool need_statistical = false)
            : option(op)
            , need_statistical(need_statistical){
//            db = new DataBase(op.loop_counts);
            ComputingDevice compute_device(op.device,op.id);
            bench = std::make_shared<Workbench>(compute_device);

            bench->runtime().set_computing_thread_number(option.num_threads);
            if(option.power_mode != -1){
                bench->set_cpu_power_mode((CpuEnable::CpuPowerMode)option.power_mode);
            }
        }

        BenchMark(const BenchMark& other)
            : option(other.option)
            ,need_statistical(other.need_statistical){
//            this->db = new DataBase(*other.db);
        }

        ~BenchMark(){
//            if(db != nullptr){
//                delete db;
//                db = nullptr;
//            }
        }

        void benchmark(const std::string name, const std::string path, const std::initializer_list<int>& input_shape){
            srand(time(NULL));
            DataBase* db = new DataBase(option.loop_counts + 1);

            std::shared_ptr<Module> m = std::make_shared<Module>();
            m = Module::Load(path);
            ComputingDevice device(option.device, option.id);

            time_point start, end;
            Hook hook;
            ctx::bind<Hook> bind_hook(hook);
            hook.before_run([&](const Hook::StructBeforeRun & before_run)->void {
                start = system_clock::now();
            });
            hook.after_run([&](const Hook::StructAfterRun & after_run)->void {
                end =  system_clock::now();
                auto duration = std::chrono::duration_cast<microseconds>(end - start);
                double time = duration.count() / 1000.0f;
                auto _op = after_run.op->op();
                auto name = after_run.op->name();
                DataBase::DataS data(_op, name, time);
                db->add(data);
            });

//            Workbench::shared bench;
//            bench = Workbench::Load(m, device, option.pack_option);
//            bench->runtime().set_computing_thread_number(option.num_threads);
//            if(option.power_mode != -1){
//                bench->set_cpu_power_mode((CpuEnable::CpuPowerMode)option.power_mode);
//            }
            bench->setup(bench->compile(m, option.compile_option));

            Shape shape = std::vector<int>(input_shape.begin(), input_shape.end());
            Tensor input_param(FLOAT32, shape);
            for (int i = 0; i < input_param.count(); i++){
                input_param.data<float>()[i] = rand() % 100 + 100;
            }
            bench->input(m->inputs()[0].bubble().name(), input_param);

            //warm up
            bench->run();
            db->index_inch();

            float count_time = 0.f;
            float max_time = 0.f,min_time = FLT_MAX;
            for (size_t i = 0; i < option.loop_counts; i++)
            {
                {
                    time_point start;
                    time_point end;
                    start = std::chrono::system_clock::now();

                    bench->run();

                    end = std::chrono::system_clock::now();
                    std::chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                    float  current_time= duration.count() / 1000.0;
                    count_time += current_time;
                    max_time = std::max(max_time, current_time);
                    min_time = std::min(min_time, current_time);
                }
                db->index_inch();
            }

            if(need_statistical){
                db->analyze();
                db->log(std::cout);
            }

            std::cout << "Net: " << name
            << " ,input:[ " << shape[0] << "," << shape[1] << "," <<  shape[2] << "," << shape[3] << "]"
            << " ,max: " << max_time << "ms"
            << " ,min: " << min_time << "ms"
            " ,avg: " << count_time / option.loop_counts << "ms"
            << std::endl;

            delete db;
        }

    private:
//        DataBase* db = nullptr;
        Option option;
        Workbench::shared bench;
        bool need_statistical = false;
    };

}





#endif //TENSORSTACK_BENCH_H
