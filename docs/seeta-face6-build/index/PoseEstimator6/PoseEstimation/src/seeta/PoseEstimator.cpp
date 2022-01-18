#include "seeta/PoseEstimator.h"

#include <orz/utils/log.h>

#include "seeta/CommonStruct.h"
#include "seeta/ImageProcess.h"

#include <api/cpp/tensorstack.h>
#include <api/cpp/module.h>

#include <orz/io/jug/jug.h>
#include <orz/io/i.h>
#include <orz/io/dir.h>
#include <orz/codec/json.h>
#include <fstream>

#ifdef SEETA_MODEL_ENCRYPT
#include "SeetaLANLock.h"
#include "hidden/SeetaLockFunction.h"
#include "hidden/SeetaLockVerifyLAN.h"
#include "orz/io/stream/filestream.h"
#endif

#include "model_helper.h"

#define VER_HEAD(x) #x "."
#define VER_TAIL(x) #x
#define GENERATE_VER(seq) FUN_MAJOR seq
#define FUN_MAJOR(x) VER_HEAD(x) FUN_MINOR
#define FUN_MINOR(x) VER_HEAD(x) FUN_SINOR
#define FUN_SINOR(x) VER_TAIL(x)

#define LIBRARY_VERSION GENERATE_VER( \
                                      (SEETA_POSE_ESTIMATOR_MAJOR_VERSION) \
                                      (SEETA_POSE_ESTIMATOR_MINOR_VERSION) \
                                      (SEETA_POSE_ESTIMATOR_SINOR_VERSION))

#define LIBRARY_NAME "PoseEstimator"

#define LOG_HEAD LIBRARY_NAME "(" LIBRARY_VERSION "): "

#include <orz/io/i.h>
#include <map>
#if SEETA_LOCK_SDK
    #include <lock/macro.h>
#endif

namespace seeta
{
    namespace v6
    {
        using namespace ts::api;

        static int CLAMP(int a, int min, int max)
        {
            return std::max(min, std::min(max, a));
        }


        static SeetaRect V6toV5( const SeetaSize &limit, const SeetaRect &face )
        {
            /**
             * INFO: width scale: 1.1311
             * INFO: height scale: 1.13779
             * INFO: x shift: -0.0683691
             * INFO: y shift: -0.060302
             */
            float width_scale = 1.1311;
            float height_scale = 1.13779;
            float x_shift = -0.0683691;
            float y_shift = -0.060302;
            SeetaRect rect = face;

            rect.x += int( x_shift * rect.width );
            rect.y += int( y_shift * rect.height );
            rect.width = int( rect.width * width_scale );
            rect.height = int( rect.height * height_scale );
            int x1 = CLAMP( rect.x, 0, limit.width - 1 );
            int y1 = CLAMP( rect.y, 0, limit.height - 1 );
            int x2 = CLAMP( rect.x + rect.width - 1, 0, limit.width - 1 );
            int y2 = CLAMP( rect.y + rect.height - 1, 0, limit.height - 1 );
            int w = x2 - x1 + 1;
            int h = y2 - y1 + 1;

            rect.x = x1;
            rect.y = y1;
            rect.width = w;
            rect.height = h;

            return rect;
        }



        static std::string read_txt_file( std::ifstream &in )
        {
            std::ostringstream tmp;
            tmp << in.rdbuf();
            return tmp.str();
        }
        static orz::jug read_jug_from_json_or_sta( const std::string &filename )
        {
            std::ifstream ifs( filename, std::ios::binary );
            int32_t mark;
            ifs.read( reinterpret_cast<char *>( &mark ), 4 );

            try
            {
                if( mark == orz::STA_MASK )
                {
                    return orz::jug_read( ifs );
                }
                else
                {
                    std::string json = read_txt_file( ifs );
                    return orz::json2jug( json );
                }
            }
            catch( const orz::Exception & )
            {
                ORZ_LOG( orz::ERROR ) << "Model must be sta or json file, given: " << filename << orz::crash;
                return orz::jug();
            }
        }

        static Module parse_tsm_module( const orz::jug &model, const std::string &root )
        {
            if( model.valid( orz::Piece::BINARY ) )
            {
                auto binary = model.to_binary();
                BufferReader reader( binary.data(), binary.size() );
                return Module::Load( reader );
            }
            else
                if( model.valid( orz::Piece::STRING ) )
                {
                    auto commands = orz::Split( model.to_string(), '@', 3 );
                    if( commands.size() != 3 || !commands[0].empty() || commands[1] != "file" )
                    {
                        ORZ_LOG( orz::ERROR ) << R"(Model: /backbone/tsm must be "@file@..." or "@binary@...")" << orz::crash;
                    }
                    std::string path = root.empty() ? commands[2] : orz::Join( {root, commands[2]}, orz::FileSeparator() );
                    return Module::Load( path );
                }
                else
                {
                    ORZ_LOG( orz::ERROR ) << R"(Model: /backbone/tsm must be "@file@..." or "@binary@...")" << orz::crash;
                }
            return Module();
        }

        struct ModelParam
        {
            ModelParam() = default;

            std::vector<orz::jug> pre_processor;

            struct
            {
                orz::jug tsm;
            } backbone;

            struct
            {
                bool normalize = true;
                int sqrt_times = 0;
            } post_processor;

            struct
            {
                float threshold = 0.05;
                struct
                {
                    std::string format = "HWC";
                    int height = 256;
                    int width = 256;
                    int channels = 3;
                } input;
                struct
                {
                    int size = 256;
                } output;
                orz::jug compare;   // an op like {"op": "dot"}, could be invalid
                orz::jug similarity;    // an op like {"op": "sigmoid", "params": [3.0, 7.0]} or {"op", "none"}, could be invalid
            } global;

            static bool to_bool( const orz::jug &jug ) {
                return jug.to_bool();
            }

            static std::vector<int> to_int_list( const orz::jug &jug ) {
                if( jug.invalid( orz::Piece::LIST ) ) throw orz::Exception( "jug must be list" );
                std::vector<int> list( jug.size() );
                for( size_t i = 0; i < list.size(); ++i ) {
                    list[i] = jug[i].to_int();
                }
                return std::move( list );
            }
            static std::vector<std::vector<int>> to_int_list_list( const orz::jug &jug ) {
                if( jug.invalid( orz::Piece::LIST ) ) throw orz::Exception( "jug must be list" );
                std::vector<std::vector<int>> list( jug.size() );
                for( size_t i = 0; i < list.size(); ++i ) {
                    list[i] = to_int_list( jug[i] );
                }
                return std::move( list );
            }
            static std::vector<float> to_float_list( const orz::jug &jug ) {
                if( jug.invalid( orz::Piece::LIST ) ) throw orz::Exception( "jug must be list" );
                std::vector<float> list( jug.size() );
                for( size_t i = 0; i < list.size(); ++i ) {
                    list[i] = jug[i].to_float();
                }
                return std::move( list );
            }

            static std::vector<std::vector<float>> to_float_list_list( const orz::jug &jug ) {
                if( jug.invalid( orz::Piece::LIST ) ) throw orz::Exception( "jug must be list" );
                std::vector<std::vector<float>> list( jug.size() );
                for( size_t i = 0; i < list.size(); ++i ) {
                    list[i] = to_float_list( jug[i] );
                }
                return std::move( list );
            }
        };


        ModelParam parse_model( const orz::jug &model )
        {
            ModelParam param;

            if( model.invalid( orz::Piece::DICT ) ) ORZ_LOG( orz::ERROR ) << "Model: / must be dict" << orz::crash;

            auto pre_processor = model["pre_processor"];
            auto backbone = model["backbone"];
            auto post_processor = model["post_processor"];
            auto global = model["global"];

            if( pre_processor.valid() )
            {
                if( pre_processor.valid( orz::Piece::LIST ) )
                {
                    auto size = pre_processor.size();
                    for( decltype( size ) i = 0; i < size; ++i )
                    {
                        param.pre_processor.emplace_back( pre_processor[i] );
                    }
                }
                else
                {
                    ORZ_LOG( orz::ERROR ) << "Model: /pre_processor must be list" << orz::crash;
                }
            }

            if( backbone.valid( orz::Piece::DICT ) )
            {
                auto tsm = backbone["tsm"];
                if( tsm.invalid() )
                {
                    ORZ_LOG( orz::ERROR ) << R"(Model: /backbone/tsm must be "@file@..." or "@binary@...")" << orz::crash;
                }
                param.backbone.tsm = tsm;
            }
            else
            {
                ORZ_LOG( orz::ERROR ) << "Model: /backbone must be dict" << orz::crash;
            }

            if( post_processor.valid() )
            {
                if( post_processor.valid( orz::Piece::DICT ) )
                {
                    param.post_processor.normalize = orz::jug_get<bool>( post_processor["normalize"], true );
                    if( !param.post_processor.normalize )
                    {
                        ORZ_LOG( orz::ERROR ) << "Model: /post_processor/normalize must be true" << orz::crash;
                    }
                    param.post_processor.sqrt_times = orz::jug_get<int>( post_processor["sqrt_times"], param.post_processor.sqrt_times );
                }
                else
                {
                    ORZ_LOG( orz::ERROR ) << "Model: /post_processor must be dict" << orz::crash;
                }
            }

            return param;
        }

        Device to_ts_device( const seeta::ModelSetting &setting )
        {
            switch( setting.get_device() )
            {
                case seeta::ModelSetting::Device::AUTO:
                    return Device( "cpu" );
                case seeta::ModelSetting::Device::CPU:
                    return Device( "cpu" );
                case seeta::ModelSetting::Device::GPU:
                    return Device( "gpu", setting.id );
                default:
                    return Device( "cpu" );
            }
        }


        static void build_filter( ImageFilter &filter, const std::vector<orz::jug> &pre_processor )
        {
            filter.clear();
            for( size_t i = 0; i < pre_processor.size(); ++i )
            {
                auto &processor = pre_processor[i];
                if( processor.invalid( orz::Piece::DICT ) )
                {
                    ORZ_LOG( orz::ERROR ) << "Model: the " << i << "-th processor \"" << processor << "\" should be dict" << orz::crash;
                }
                auto op = orz::jug_get<std::string>( processor["op"], "" );
                if( op.empty() )
                {
                    ORZ_LOG( orz::ERROR ) << R"(Model: processor should be set like {"op": "to_float"}.)" << orz::crash;
                }
                if( op == "to_float" )
                {
                    filter.to_float();
                }
                else
                    if( op == "to_chw" )
                    {
                        filter.to_chw();
                    }
                    else
                        if( op == "sub_mean" )
                        {
                            std::vector<float> mean;
                            try
                            {
                                mean = ModelParam::to_float_list( processor["mean"] );
                            }
                            catch( ... ) {}
                            if( mean.empty() )
                            {
                                ORZ_LOG( orz::ERROR ) << R"(Model: processor "sub_mean" must set "mean" like "{"op": "sub_mean", "mean": [104, 117, 123]}")" << orz::crash;
                            }
                            filter.sub_mean( mean );
                        }
                        else
                            if( op == "center_crop" )
                            {
                                std::vector<int> size;
                                try
                                {
                                    size = ModelParam::to_int_list( processor["size"] );
                                }
                                catch( ... ) {}
                                if( size.empty() )
                                {
                                    ORZ_LOG( orz::ERROR ) << R"(Model: processor "center_crop" must set "mean" like "{"op": "center_crop", "size": [248, 248]}")" << orz::crash;
                                }
                                if( size.size() == 1 )
                                {
                                    filter.center_crop( size[0] );
                                }
                                else
                                {
                                    filter.center_crop( size[0], size[1] );
                                }
                            }
                            else
                                if( op == "resize" )
                                {
                                    std::vector<int> size;
                                    try
                                    {
                                        size = ModelParam::to_int_list( processor["size"] );
                                    }
                                    catch( ... ) {}
                                    if( size.empty() )
                                    {
                                        ORZ_LOG( orz::ERROR ) << R"(Model: processor "resize" must set "mean" like "{"op": "resize", "size": [248, 248]}")" << orz::crash;
                                    }
                                    if( size.size() == 1 )
                                    {
                                        filter.resize( size[0] );
                                    }
                                    else
                                    {
                                        filter.resize( size[0], size[1] );
                                    }
                                }
                                else
                                    if( op == "prewhiten" )
                                    {
                                        filter.prewhiten();
                                    }
                                    else
                                        if( op == "channel_swap" )
                                        {
                                            std::vector<int> shuffle;
                                            try
                                            {
                                                shuffle = ModelParam::to_int_list( processor["shuffle"] );
                                            }
                                            catch( ... ) {}
                                            if( shuffle.size() != 3 )
                                            {
                                                ORZ_LOG( orz::ERROR ) << R"(Model: processor "resize" must set "mean" like "{"op": "channel_swap", "shuffle": [2, 1, 0]}")" << orz::crash;
                                            }
                                            filter.channel_swap( shuffle );
                                        }
                                        else
                                        {
                                            ORZ_LOG( orz::ERROR ) << "Model: processor \"" << processor << "\" not supported." << orz::crash;
                                        }
            }
        }

        static std::string to_string( const std::vector<int> &shape )
        {
            std::ostringstream oss;
            oss << "[";
            for( size_t i = 0; i < shape.size(); ++i )
            {
                if( i ) oss << ", ";
                oss << shape[i];
            }
            oss << "]";
            return oss.str();
            ( void )( to_string );
        }

        static float raw2degree( float raw )
        {
            return float( ( 1.0 / ( 1.0 + std::exp( -raw ) ) ) * 180.0 - 90.0 );
        }

        class PoseEstimator::Implement {
        public:
            Implement( const seeta::ModelSetting &setting ) {
                auto &model = setting.get_model();
                if( model.size() != 1 ) {
                    ORZ_LOG( orz::ERROR ) << "Must have 1 model. Current appended model num is "			<<model.size()<<": ";
		    for(int i = 0;i < model.size(); ++i)
		    {
			ORZ_LOG(orz::ERROR)<<model[i]<<" ";
                    }
		   
		 ORZ_LOG(orz::ERROR)<<orz::crash;
                }

                auto jug = get_model_jug( model[0].c_str() );

                auto param = parse_model( jug );

                // parse tsm module
                std::string root = orz::cut_path_tail( model[0] );
                auto tsm = parse_tsm_module( param.backbone.tsm, root );
                // add image filter
                auto device = to_ts_device( setting );
                auto bench = Workbench::Load( tsm, device );
                // ts_Workbench_setup_device(bench.get_raw());
                ImageFilter filter( device );

                build_filter( filter, param.pre_processor );
                bench.bind_filter( 0, filter );

                this->m_param = param;
                this->m_bench = bench;

            }


            bool Estimate( const SeetaImageData &image, const SeetaRect &info, float &yaw, float &pitch, float &roll ) {
                #ifdef SEETA_CHECK_AUTO_FUNCID
                SEETA_CHECK_AUTO_FUNCID( "PoseEstimation" );
                #endif

                yaw = pitch = roll = 0;
                if( !image.data || image.channels != 3 ) {
                    std::cout << "----------" << std::endl;
                    return false;
                }


                SeetaSize facesize;
                facesize.width = image.width;
                facesize.height =  image.height;
                SeetaRect facerect = V6toV5(facesize, info); 


                seeta::Size size( 90, 90 );
                seeta::Rect rect( facerect.x, facerect.y, facerect.width, facerect.height );

                
                seeta::Image cropped_face = seeta::crop_resize( image, rect, size );

                auto tensor = tensor::build( UINT8, {1, cropped_face.height(), cropped_face.width(), cropped_face.channels()}, cropped_face.data() );
                m_bench.input( 0, tensor );
                m_bench.run();
                auto output = tensor::cast( FLOAT32, m_bench.output( 0 ) );

                if( output.count() != 3 ) {
                    ORZ_LOG( orz::ERROR ) << "Extracted features size must be 3 vs. "
                                            << output.count() << " given.";
                    return false;
                }


                float value[3];
                std::memcpy( value, output.data(), 3 * sizeof( float ) );
                #ifdef _DEBUG
                int _count = 3;
                std::cout << "LOG: Predict count: " << _count << std::endl;
                std::cout << "LOG: Predict result: ";
                for( int i = 0; i < _count; ++i ) {
                    if( i ) std::cout << ", ";
                    std::cout << value[i];
                }
                std::cout << std::endl;
                #endif  // _DEBUG

                yaw = raw2degree( value[0] );
                pitch = raw2degree( value[1] );
                roll = raw2degree( value[2] );

                return true;
            }


            void Feed( const SeetaImageData &image, const SeetaRect &face ) {
                bool bret = Estimate( image, face, m_angle_yaw, m_angle_pitch, m_angle_roll );
                if( !bret ) {
                    ORZ_LOG( orz::ERROR ) << "Estimate failed!" << orz::crash;
                }
            }


            int get_cpu_affinity() const {
                return m_cpu_affinity;
            }

            void set_cpu_affinity(int level) {
                switch (level) {
                    case 0:
                        m_bench.set_cpu_mode(CpuPowerMode::BIG_CORE);
                        break;
                    case 1:
                        m_bench.set_cpu_mode(CpuPowerMode::LITTLE_CORE);
                        break;
                    case 2:
                        m_bench.set_cpu_mode(CpuPowerMode::BALANCE);
                        break;
                    default:
                        level = -1;
                        break;
                }
                m_cpu_affinity = level;
            }

  
            void set(PoseEstimator::Property property, double value) {
                switch (property) {
                    default:
                        break;

                    case PoseEstimator::PROPERTY_NUMBER_THREADS:
                    {
                        if (value < 1) value = 1;
                        auto threads = int(value);
                        m_number_threads = threads;
                        m_bench.set_computing_thread_number(threads);
                        break;
                    }
                    case PoseEstimator::PROPERTY_ARM_CPU_MODE:
                    {
                        set_cpu_affinity(int32_t(value));
                        break;
                    }

                }
            }

            double get(PoseEstimator::Property property) const {
                switch (property) {
                    default:
                        return 0;

                    case PoseEstimator::PROPERTY_NUMBER_THREADS:
                        return m_number_threads;
                    case PoseEstimator::PROPERTY_ARM_CPU_MODE:
                        return get_cpu_affinity();

                }
            }


            float m_angle_yaw = 0;
            float m_angle_pitch = 0;
            float m_angle_roll = 0;
            ModelParam m_param;
            mutable Workbench m_bench;

            static const int32_t M_MIN_FACE_SIZE_POWER = 20;
            int32_t m_min_face_size = M_MIN_FACE_SIZE_POWER;
            int32_t m_max_image_width = 2000;
            int32_t m_max_image_height = 2000;
            int32_t m_number_threads = 4;
            int m_cpu_affinity = -1;
        };


        //////////////////////////////////////

        PoseEstimator::PoseEstimator( const seeta::ModelSetting &setting )
            : m_impl( new Implement( setting ) )
        {
        }

        PoseEstimator::~PoseEstimator()
        {
            delete m_impl;
        }


        void PoseEstimator::Feed( const SeetaImageData &image, const SeetaRect &face ) const
        {
            m_impl->Feed( image, face );
        }

        float PoseEstimator::Get( Axis axis ) const
        {
            switch( axis )
            {
                case YAW:
                    return m_impl->m_angle_yaw;
                case PITCH:
                    return m_impl->m_angle_pitch;
                case ROLL:
                    return m_impl->m_angle_roll;
                default:
                    return 0;
            }
        }

        void PoseEstimator::Estimate( const SeetaImageData &image, const SeetaRect &face, float *yaw, float *pitch,
                                      float *roll ) const
        {
            m_impl->Feed( image, face );
            if( yaw )    *yaw    = m_impl->m_angle_yaw;
            if( pitch )  *pitch  = m_impl->m_angle_pitch;
            if( roll )   *roll   = m_impl->m_angle_roll;
        }


        void PoseEstimator::set(PoseEstimator::Property property, double value) {
            m_impl->set(property, value);
        }

        double PoseEstimator::get(PoseEstimator::Property property) const {
            return m_impl->get(property);
        }



    }
}

