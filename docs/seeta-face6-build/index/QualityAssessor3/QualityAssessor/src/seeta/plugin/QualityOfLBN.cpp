#include "seeta/QualityOfLBN.h"

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


#include "seeta/model_helper.h"


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

#define LIBRARY_NAME "QualityOfLBN"

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

        static int CLAMP( int a, int min, int max )
        {
            return std::max( min, std::min( max, a ) );
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


        class QualityOfLBN::Implement {
        public:
            Implement( const seeta::ModelSetting &setting ) {
				
                auto &model = setting.get_model();
                if( model.size() != 1 ) {
                    ORZ_LOG( orz::ERROR ) << "Must have 1 model. Current appended model num is "            << model.size() << ": ";
                    for( int i = 0; i < model.size(); ++i ) {
                        ORZ_LOG( orz::ERROR ) << model[i] << " ";
                    }

                    ORZ_LOG( orz::ERROR ) << orz::crash;
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

            template<typename T>
            static int argmax( T *data, int count)  {
                if( count <= 0 ) {
                    return -1;
                }

                int ret = 0;

                
                for( int i = 1; i < count; i++ ) {
                    if( data[ret] < data[i] ) {
                        ret = i;
                    }
                }
                /* 
                std::cout << "-------data:" << "size:" << count << " :";
                for(int i=0; i<count; i++)
                {
                    std::cout << data[i] << ",";
                }
                std::cout << std::endl;
                */ 
                return ret;
            }

            static void fillimage(seeta::Image &dst, const seeta::Point &point, const seeta::Image &src, const SeetaRect &rect)
        {
                if (dst.channels() != src.channels())
                {
                     throw std::logic_error(str("Can not file image with mismatch channels ", src.channels(), " vs ", dst.channels()));
                }

                if (dst.width() - point.x < rect.width || dst.height() - point.y < rect.height)
                {
                    throw std::logic_error(str("dst valid area less than copy area"));
                }

                int copy_size = rect.width * src.channels();
                if (copy_size <= 0) return;
                int dst_step = dst.width() * dst.channels();
                int src_step = src.width() * src.channels();
                auto *dst_ptr = &dst.data(point.y, point.x, 0);
                const auto *src_ptr = &src.data(rect.y, rect.x, 0);
                for (int i = 0; i < rect.height; ++i)
                {
                    CopyData(dst_ptr, src_ptr, copy_size);
                    dst_ptr += dst_step;
                    src_ptr += src_step;
                }
            }


            static const Image crop_image( const SeetaImageData &image, const SeetaPointF *points, float multiple ) 
            {
                int mid_top_x = ( points[22].x  + points[21].x ) * 0.5;
                int mid_top_y = ( points[22].y + points[21].y ) * 0.5;
                int width = points[16].x - points[0].x;
                int mid_bottom_x = points[8].x;
                int mid_bottom_y = points[8].y;

                int x_mid = ( mid_top_x + mid_bottom_x ) * 0.5;
                int y_mid = ( mid_top_y + mid_bottom_y ) * 0.5;

                int roi_left = x_mid - width * ( multiple / 2 );
                int roi_top = y_mid - width * ( multiple / 2 );
                int roi_right = x_mid + width * ( multiple / 2 );
                int roi_bottom = y_mid + width * ( multiple / 2 );

                int cut_x = roi_left < 0 ? 0 : roi_left;
                int cut_y = roi_top < 0 ? 0 : roi_top;

                int cut_right = roi_right < image.width ? roi_right : image.width;
                int cut_bottom = roi_bottom < image.height ? roi_bottom : image.height;

                int width_roi = cut_right - cut_x;
                int height_roi = cut_bottom - cut_y;

                //std::cout << "222 cut_x:" << cut_x << ",cut_y:" << cut_y << ",width:" << width_roi << ",height:" << height_roi << std::endl;
                Rect rect(cut_x, cut_y, width_roi, height_roi);
                //const Image cutimage = crop(image, rect);

                //cv::Rect rect(cut_x, cut_y, width_roi, height_roi);
                //auto image_roi = image(rect);

                int left = x_mid - width * ( multiple / 2 ) < 0 ? std::abs( x_mid - width * ( multiple / 2 ) ) : 0;
                int top = y_mid - width * ( multiple / 2 ) < 0 ? std::abs( y_mid - width * ( multiple / 2 ) ) : 0;
                int right = x_mid + width * ( multiple / 2 ) <= image.width ? 0 : x_mid + width * ( multiple / 2 ) - image.width;
                int bottom = y_mid + width * ( multiple / 2 ) < image.height ? 0 : y_mid + width * ( multiple / 2 ) - image.height;
                //std::cout << "222 left:" << left << ",top:" << top << ",right:" << right << ",bottom:" << bottom << std::endl;
                //std::cout << "222 width_roi:" << width_roi << ",height_roi:" << height_roi << std::endl;
                Image result(width_roi + left + right, height_roi + top + bottom, image.channels);
                memset(result.data(), 0, result.count() * sizeof(Image::Datum));

                Point point(left, top);

                fillimage(result, point, image, rect); 
                //fill(result, point, cutimage);
                return std::move(result);
            }


            bool Detect( const SeetaImageData &image, const SeetaPointF *points, int *light, int *blur, int *noise ) {
                #ifdef SEETA_CHECK_AUTO_FUNCID
                SEETA_CHECK_AUTO_FUNCID( "QualityOfLBN" );
                #endif

                if( !image.data || image.channels != 3 ) {
                    return false;
                }

                seeta::Size size( 256, 256 );

                seeta::Image tmpimage= crop_image(image, points, 1.5);
                seeta::Image cropped_face = resize(tmpimage, size);

                auto tensor = tensor::build( UINT8, {1, cropped_face.height(), cropped_face.width(), cropped_face.channels()}, cropped_face.data() );

                m_bench.input( 0, tensor );
                m_bench.run();

                if( m_bench.output_count() != 3 ) {
                    ORZ_LOG( orz::ERROR ) << "tensorstack output_count != 3" << orz::crash;
                    return false;
                }
                auto light_output = tensor::cast( FLOAT32, m_bench.output( 0 ) );
                auto blur_output = tensor::cast( FLOAT32, m_bench.output( 1 ) );
                auto noise_output = tensor::cast( FLOAT32, m_bench.output( 2 ) );

                assert(light_output.count() > 0);
                assert(blur_output.count() > 0);
                assert(noise_output.count() > 0);

                

                float *ptr = light_output.data<float>();
                if(ptr[0] >= m_light_thresh) 
                {
                    *light = 0;
                }else
                {
                    *light = 1;
                }

                ptr = blur_output.data<float>();
                if(ptr[0] >= m_blur_thresh) 
                {
                    *blur = 0;
                }else
                {
                    *blur = 1;
                }

                //std::cout << "blur:" << ptr[0] << std::endl;
                ptr = noise_output.data<float>();
                if(ptr[0] >= m_noise_thresh) 
                {
                    *noise = 0;
                }else
                {
                    *noise = 1;
                }

                return true;
            }

            int get_cpu_affinity() const {
                return m_cpu_affinity;
            }

            void set_cpu_affinity( int level ) {
                switch( level ) {
                    case 0:
                        m_bench.set_cpu_mode( CpuPowerMode::BIG_CORE );
                        break;
                    case 1:
                        m_bench.set_cpu_mode( CpuPowerMode::LITTLE_CORE );
                        break;
                    case 2:
                        m_bench.set_cpu_mode( CpuPowerMode::BALANCE );
                        break;
                    default:
                        level = -1;
                        break;
                }
                m_cpu_affinity = level;
            }


            void set( QualityOfLBN::Property property, double value ) {
                switch( property ) {
                    default:
                        break;
                    case QualityOfLBN::PROPERTY_NUMBER_THREADS: {
                        if( value < 1 ) value = 1;
                        auto threads = int( value );
                        m_number_threads = threads;
                        m_bench.set_computing_thread_number( threads );
                        break;
                    }
                    case QualityOfLBN::PROPERTY_ARM_CPU_MODE: {
                        set_cpu_affinity( int32_t( value ) );
                        break;
                    }

                    case QualityOfLBN::PROPERTY_LIGHT_THRESH: {
                         m_light_thresh = float(value);
                         break;
                    }

                    case QualityOfLBN::PROPERTY_BLUR_THRESH: {
                         m_blur_thresh = float(value);
                         break;
                    }
                    case QualityOfLBN::PROPERTY_NOISE_THRESH: {
                         m_noise_thresh = float(value);
                         break;
                    }
                }
            }

            double get( QualityOfLBN::Property property ) const {
                switch( property ) {
                    default:
                        return 0;
                    case QualityOfLBN::PROPERTY_NUMBER_THREADS:
                        return m_number_threads;
                    case QualityOfLBN::PROPERTY_ARM_CPU_MODE:
                        return get_cpu_affinity();

                    case QualityOfLBN::PROPERTY_LIGHT_THRESH:
                        return m_light_thresh;
                    case QualityOfLBN::PROPERTY_BLUR_THRESH:
                        return m_blur_thresh;
                    case QualityOfLBN::PROPERTY_NOISE_THRESH:
                        return m_noise_thresh;

                }
            }


            ModelParam m_param;
            mutable Workbench m_bench;

            static const int32_t M_MIN_FACE_SIZE_POWER = 20;
            int32_t m_min_face_size = M_MIN_FACE_SIZE_POWER;
            int32_t m_max_image_width = 2000;
            int32_t m_max_image_height = 2000;
            int32_t m_number_threads = 4;
            int m_cpu_affinity = -1;

            float m_light_thresh = 0.5;
            float m_blur_thresh = 0.80;
            float m_noise_thresh = 0.5;

        };


        //////////////////////////////////////

        QualityOfLBN::QualityOfLBN( const seeta::ModelSetting &setting )
            : m_impl( new Implement( setting ) )
        {
        }

        QualityOfLBN::~QualityOfLBN()
        {
            delete m_impl;
        }

        void QualityOfLBN::Detect( const SeetaImageData &image, const SeetaPointF *points, int *light, int *blur,
                                 int *noise ) const
        {
            int local_light, local_blur, local_noise;
            local_light = local_blur = local_noise = 0;

            m_impl->Detect( image, points, &local_light, &local_blur, &local_noise );

            if( local_light < 0 ) local_light = 0;
            if( local_blur < 0 ) local_blur = 0;
            if( local_noise < 0 ) local_noise = 0;

            if( light )    *light    = local_light;
            if( blur )  *blur  = local_blur;
            if( noise )   *noise   = local_noise;
        }


        void QualityOfLBN::set( QualityOfLBN::Property property, double value )
        {
            m_impl->set( property, value );
        }

        double QualityOfLBN::get( QualityOfLBN::Property property ) const
        {
            return m_impl->get( property );
        }

    }
}

