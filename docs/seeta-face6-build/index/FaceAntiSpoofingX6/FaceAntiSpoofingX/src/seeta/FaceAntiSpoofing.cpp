#include "seeta/FaceAntiSpoofing.h"

#include <orz/utils/log.h>

#include <orz/io/jug/jug.h>
#include <orz/io/i.h>
#include <orz/io/dir.h>
#include <orz/codec/json.h>
#include <orz/sync/shotgun.h>
#include <orz/tools/ctxmgr_lite.h>

#include <seeta/common_alignment.h>
#include "seeta/CommonStruct.h"
#include "seeta/ImageProcess.h"

#include <iomanip>

#include <string>

#include <queue>

#include "anchor.h"
#include "CLabeledBox.h"

#include <api/cpp/tensorstack.h>
#include <api/cpp/module.h>

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
                                      (SEETA_FACE_ANTI_SPOOFING_MAJOR_VERSION) \
                                      (SEETA_FACE_ANTI_SPOOFING_MINOR_VERSION) \
                                      (SEETA_FACE_ANTI_SPOOFING_SINOR_VERSION))

#define LIBRARY_NAME "FaceAntiSpoofing"

#define LOG_HEAD LIBRARY_NAME "(" LIBRARY_VERSION "): "

namespace seeta
{
    namespace v6
    {

        ////////////////////////////////////////
        using namespace ts::api;
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


        /////////////////////////////////////////
        class BoxDetector {
        public:
            explicit BoxDetector( const seeta::ModelSetting &setting );
            ~BoxDetector();


            SeetaLabeledBoxArray Detect( const SeetaImageData &image ) const;

            void SetThreshold( float threshold = 0.3 );

            float GetThreshold() const;

            void SetNMS( float nms = 0.6 );

            float GetNMS() const;

            void SetVideoStable( bool stable = true );

            bool GetVideoStable() const;

        private:
            BoxDetector( const BoxDetector &other ) = delete;
            const BoxDetector &operator=( const BoxDetector &other ) = delete;

        private:
            class Implement;
            Implement *m_impl;
        };

        class BoxDetector::Implement {
        public:
            using self = Implement;

            Implement( const seeta::ModelSetting &setting ) {
                auto &model = setting.get_model();
                if( model.size() != 1 ) {
                    ORZ_LOG( orz::ERROR ) << "Must have 1 model." << orz::crash;
                }
                auto jug = get_model_jug( model[0].c_str() );

                auto param = parse_model( jug );

                // parse tsm module
                std::string root = orz::cut_path_tail( model[0] );
                auto tsm = parse_tsm_module( param.backbone.tsm, root );

                ////////////////////////
                ///////////////////

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

            static float IoU( const Rect &rect1, const Rect &rect2 ) {
                auto x1 = rect1.x;
                auto y1 = rect1.y;
                auto z1 = rect1.x + rect1.width - 1;
                auto k1 = rect1.y + rect1.height - 1;
                auto x2 = rect2.x;
                auto y2 = rect2.y;
                auto z2 = rect2.x + rect2.width - 1;
                auto k2 = rect2.y + rect2.height - 1;
                auto xOverlap = std::max( 0, std::min( z1, z2 ) - std::max( x1, x2 ) + 1 );
                auto yOverlap = std::max( 0, std::min( k1, k2 ) - std::max( y1, y2 ) + 1 );
                auto I = xOverlap * yOverlap;
                auto U = rect1.width * rect1.height + rect2.width * rect2.height - I;
                return float( I ) / U;
            }

            enum NMSType
            {
                nms_local,
                nms_global
            };
            static std::vector<SeetaLabeledBox> NMS( const std::vector<SeetaLabeledBox> &box_list, float threshold ) {
                auto list = box_list;
                std::sort( list.begin(), list.end(), []( const SeetaLabeledBox & info1, const SeetaLabeledBox & info2 ) {
                    if( info1.score == info2.score ) {
                        return info1.pos.width * info1.pos.height > info2.pos.width * info2.pos.height;
                    }
                    else return info1.score > info2.score;
                } );

                std::vector<bool> flag( list.size(), false );
                for( size_t i = 0; i < list.size(); ++i ) {
                    if( flag[i] ) continue;
                    for( size_t j = i + 1; j < list.size(); ++j ) {
                        if( IoU( list[i].pos, list[j].pos ) > threshold ) flag[j] = true;
                    }
                }
                std::vector<SeetaLabeledBox> result;
                for( size_t i = 0; i < list.size(); ++i ) {
                    if( !flag[i] ) result.push_back( list[i] );
                }
                return result;
            }

            static void softmax( float *arr, size_t size ) {
                float sum = 0;
                for( size_t i = 1; i < size; ++i ) {
                    arr[i] = std::exp( arr[i] );
                    sum += arr[i];
                }
                for( size_t i = 1; i < size; ++i ) {
                    arr[i] /= sum;
                }
            }

            static size_t argmax( const float *arr, size_t size ) {
                if( size == 0 ) return 0;
                auto flag = arr[0];
                size_t max = 0;
                for( size_t i = 1; i < size; ++i ) {
                    if( arr[i] > flag ) {
                        flag = arr[i];
                        max = i;
                    }
                }
                return max;
            }

            static std::array<float, 4> decode_box(
                std::array<float, 4> anchor,
                std::array<float, 4> box_encoding ) {
                std::swap( anchor[0], anchor[1] );
                std::swap( anchor[2], anchor[3] );
                std::swap( box_encoding[0], box_encoding[1] );
                std::swap( box_encoding[2], box_encoding[3] );

                auto width = anchor[2] - anchor[0];
                auto height = anchor[3] - anchor[1];
                auto ctr_x = anchor[0] + 0.5f * width;
                auto ctr_y = anchor[1] + 0.5f * height;

                auto pred_ctr_x = box_encoding[0] * 0.1f * width + ctr_x;
                auto  pred_ctr_y = box_encoding[1] * 0.1f * height + ctr_y;
                auto  pred_w = std::exp( box_encoding[2] * 0.2f ) * width;
                auto  pred_h = std::exp( box_encoding[3] * 0.2f ) * height;

                std::array<float, 4> region =
                {
                    pred_ctr_x - 0.5f * pred_w,
                    pred_ctr_y - 0.5f * pred_h,
                    pred_ctr_x + 0.5f * pred_w,
                    pred_ctr_y + 0.5f * pred_h
                };

                return region;
            }

            template <typename T>
            static T clamp( const T &value, const T &min, const T &max ) {
                return std::min<int>( std::max<int>( value, min ), max );
            }

            static SeetaRect fix( const SeetaRect &rect, int width, int height ) {
                int x0 = rect.x;
                int y0 = rect.y;
                int x1 = rect.x + rect.width;
                int y1 = rect.y + rect.height;
                x0 = clamp( x0, 0, width );
                y0 = clamp( y0, 0, height );
                x1 = clamp( x1, 0, width );
                y1 = clamp( y1, 0, height );
                seeta::Rect fixed( x0, y0, x1 - x0, y1 - y0 );
                return fixed;
            }

            std::vector<SeetaLabeledBox> Detect( const Image &image ) const {
                // TODO: finish detection
                std::vector<SeetaLabeledBox> result;

                seeta::Blob<float> fimage = seeta::resize( image, seeta::Size( 300, 300 ) );

                fimage.for_each( []( float & x ) {
                    x *= 0.00784313771874f;
                } );
                fimage.for_each( []( float & x ) {
                    x -= 1.0f;
                } );
               
                auto tensor = tensor::build(FLOAT32, fimage.shape(), fimage.data() );
                m_bench.input( 0, tensor );
                m_bench.run();
               
                auto box_encodings = tensor::cast( FLOAT32, m_bench.output( 0 ) );
                auto class_predictions = tensor::cast( FLOAT32, m_bench.output( 1 ) );
                auto size = class_predictions.size( 1 );
                //std::cout << "class_predictions.shape(1):" << size <<  std::endl;

                float *pbox_encodings= box_encodings.data<float>();
                float *pclass_predictions = class_predictions.data<float>();
                
                for( int i = 0; i < size; ++i ) { 

                    size_t label = argmax( pclass_predictions + i * class_predictions.size( 2 ) *  class_predictions.size( 3 )  + 1, 2 ) + 1;
                    float score = class_predictions.data<float>( i * class_predictions.size( 2 ) * class_predictions.size( 3 ) + int( label ) * class_predictions.size( 3 ) );
                    
                    if( score < class_threshold ) continue;

                    std::array<float, 4> box_encoding;

                    box_encoding[0] = box_encodings.data<float>( i * box_encodings.size( 2 ) * box_encodings.size( 3 ) + 0 );
                    box_encoding[1] = box_encodings.data<float>( i * box_encodings.size( 2 ) * box_encodings.size( 3 ) + 1 );
                    box_encoding[2] = box_encodings.data<float>( i * box_encodings.size( 2 ) * box_encodings.size( 3 ) + 2 );
                    box_encoding[3] = box_encodings.data<float>( i * box_encodings.size( 2 ) * box_encodings.size( 3 ) + 3 );

                    std::array<float, 4> anchor;
                    anchor[0] = seeta::box_detection::anchors[i][0];
                    anchor[1] = seeta::box_detection::anchors[i][1];
                    anchor[2] = seeta::box_detection::anchors[i][2];
                    anchor[3] = seeta::box_detection::anchors[i][3];
                    auto region = decode_box( anchor, box_encoding );

                    auto x = region[0];
                    auto y = region[1];
                    auto width = region[2] - region[0];
                    auto height = region[3] - region[1];

                    SeetaLabeledBox box;
                    box.label = int( label );
                    box.score = score;
                    box.pos.x = int( x * image.width() );
                    box.pos.y = int( y * image.height() );
                    box.pos.width = int( width * image.width() );
                    box.pos.height = int( height * image.height() );
                    //std::cout << "---box---label:" << box.label << ",score:" << score << ",x:" << box.pos.x << ",y:" << box.pos.y << ",width:" << box.pos.width << ",height;" << box.pos.height << std::endl;
                    box.pos = fix( box.pos, image.width(), image.height() );

                    result.push_back( box );
                }

                result = NMS( result, nms_threshold );

                return result;
            }

            static float IoM( const Rect &rect1, const Rect &rect2 ) {
                auto x1 = rect1.x;
                auto y1 = rect1.y;
                auto z1 = rect1.x + rect1.width - 1;
                auto k1 = rect1.y + rect1.height - 1;
                auto x2 = rect2.x;
                auto y2 = rect2.y;
                auto z2 = rect2.x + rect2.width - 1;
                auto k2 = rect2.y + rect2.height - 1;
                auto xOverlap = std::max( 0, std::min( z1, z2 ) - std::max( x1, x2 ) + 1 );
                auto yOverlap = std::max( 0, std::min( k1, k2 ) - std::max( y1, y2 ) + 1 );
                auto I = xOverlap * yOverlap;
                auto M = std::min( rect1.width * rect1.height, rect2.width * rect2.height );
                return float( I ) / M;
            }

            static std::vector<SeetaLabeledBox> FinalNMS( const std::vector<SeetaLabeledBox> &rect_list, float threshold ) {
                auto list = rect_list;
                std::sort( list.begin(), list.end(), []( const SeetaLabeledBox & info1, const SeetaLabeledBox & info2 ) {
                    return info1.pos.width * info1.pos.height > info2.pos.width * info2.pos.height;
                } );
                std::vector<bool> flag( list.size(), false );
                for( size_t i = 0; i < list.size(); ++i ) {
                    if( flag[i] ) continue;
                    for( size_t j = i + 1; j < list.size(); ++j ) {
                        if( IoM( list[i].pos, list[j].pos ) > threshold ) flag[j] = true;
                    }
                }
                std::vector<SeetaLabeledBox> result;
                for( size_t i = 0; i < list.size(); ++i ) {
                    if( !flag[i] ) result.push_back( list[i] );
                }
                return result;
            }

            friend class BoxDetector;
        private:

            ModelParam m_param;
            mutable Workbench m_bench;

            float class_threshold = 0.8f;
            float nms_threshold = 0.8f;

            bool stable = false;
            std::vector<SeetaLabeledBox> pre_boxs;
        };


        BoxDetector::BoxDetector( const seeta::ModelSetting &setting )
            : m_impl( new Implement( setting ) )
        {
        }

        BoxDetector::~BoxDetector()
        {
            delete m_impl;
        }

        SeetaLabeledBoxArray BoxDetector::Detect( const SeetaImageData &image ) const
        {
            SeetaLabeledBoxArray box_array = { nullptr, 0 };

            if( image.data == nullptr || image.channels <= 0 || image.width <= 0 || image.height <= 0 ) return box_array;

            std::vector<SeetaLabeledBox> boxs = m_impl->Detect( image );

            boxs = Implement::FinalNMS( boxs, 0.9f );

            std::sort( boxs.begin(), boxs.end(), []( const SeetaLabeledBox & lhs, const SeetaLabeledBox & rhs )
            {
                return lhs.pos.width * lhs.pos.height > rhs.pos.width * rhs.pos.height;
            } );

            auto &pre_boxs = m_impl->pre_boxs;

            if( m_impl->stable )
            {
                for( size_t i = 0; i < boxs.size(); i++ )
                {
                    for( size_t j = 0; j < pre_boxs.size(); j++ )
                    {
                        if( Implement::IoU( boxs[i].pos, pre_boxs[j].pos ) > 0.80 )
                        {
                            boxs[i].pos.x = pre_boxs[j].pos.x;
                            boxs[i].pos.y = pre_boxs[j].pos.y;
                            boxs[i].pos.width = pre_boxs[j].pos.width;
                            boxs[i].pos.height = pre_boxs[j].pos.height;
                        }
                        else
                            if( Implement::IoU( boxs[i].pos, pre_boxs[j].pos ) > 0.55 )
                            {
                                boxs[i].pos.x = ( boxs[i].pos.x + pre_boxs[j].pos.x ) / 2;
                                boxs[i].pos.y = ( boxs[i].pos.y + pre_boxs[j].pos.y ) / 2;
                                boxs[i].pos.width = ( boxs[i].pos.width + pre_boxs[j].pos.width ) / 2;
                                boxs[i].pos.height = ( boxs[i].pos.height + pre_boxs[j].pos.height ) / 2;
                            }
                    }
                }
            }

            pre_boxs = boxs;

            box_array.data = pre_boxs.data();
            box_array.size = static_cast<int>( pre_boxs.size() );

            return box_array;
        }
        void BoxDetector::SetThreshold( float threshold )
        {
			if (!this) return;

            m_impl->class_threshold = threshold;
        }

        float BoxDetector::GetThreshold() const
        {
			if (!this) return -1;

            return m_impl->class_threshold;
        }

        void BoxDetector::SetNMS( float nms )
        {
            this->m_impl->nms_threshold;
        }

        float BoxDetector::GetNMS() const
        {
            return m_impl->nms_threshold;
        }

        void BoxDetector::SetVideoStable( bool stable )
        {
            m_impl->stable = stable;
            if( !m_impl->stable ) m_impl->pre_boxs.clear();
        }

        bool BoxDetector::GetVideoStable() const
        {
            return m_impl->stable;
        }


        //////////////////////////////////////
        class FaceAntiSpoofingX {
        public:

            using SeetaPointF5 = SeetaPointF[5];
            FaceAntiSpoofingX( const seeta::ModelSetting &setting ) {
                auto &model = setting.get_model();
                if( model.size() != 1 ) {
                    ORZ_LOG( orz::ERROR ) << "Must have 1 model." << orz::crash;
                }
                auto jug = get_model_jug( model[0].c_str() );

                auto param = parse_model( jug );

                // parse tsm module
                std::string root = orz::cut_path_tail( model[0] );
                auto tsm = parse_tsm_module( param.backbone.tsm, root );
                // add image filter
                auto device = to_ts_device( setting );
                auto bench = Workbench::Load( tsm, device );
                ImageFilter filter( device );

                build_filter( filter, param.pre_processor );
                bench.bind_filter( 0, filter );

                this->m_param = param;
                this->m_bench = bench;

                m_gun = std::make_shared<orz::Shotgun>( 4 );

            }
            ~FaceAntiSpoofingX()
            {
            }

            void load_box_model( const seeta::ModelSetting &setting ) {
                //using self = SeetaFaceSpoofDetect;
                m_box_detetor = std::make_shared<seeta::BoxDetector>( setting );

            }

            static seeta::Image copyMakeBorder( const seeta::Image &img, int top, int bottom, int left, int right ) {
                if( top == 0 && bottom == 0 && left == 0 && right == 0 ) return img;
                seeta::Image result( img.width() + left + right, img.height() + top + bottom, img.channels() );
                seeta::fill( result, seeta::Point( left, top ), img );
                if( img.width() == 0 || img.height() == 0 ) return result;
                // left
                for( int y = top; y < top + img.height(); ++y ) {
                    for( int x = 0; x < left; ++x ) {
                        std::memcpy( &result.data( y, x, 0 ), &result.data( y, left, 0 ), result.channels() );
                    }
                }
                // right
                for( int y = top; y < top + img.height(); ++y ) {
                    for( int x = left + img.width(); x < result.width(); ++x ) {
                        std::memcpy( &result.data( y, x, 0 ), &result.data( y, left + img.width() - 1, 0 ), result.channels() );
                    }
                }
                // top
                for( int y = 0; y < top; ++y ) {
                    std::memcpy( &result.data( y, 0, 0 ), &result.data( top, 0, 0 ), result.width() * result.channels() );
                }
                // bottom
                for( int y = top + img.height(); y < result.height(); ++y ) {
                    std::memcpy( &result.data( y, 0, 0 ), &result.data( top + img.height() - 1, 0, 0 ), result.width() * result.channels() );
                }
                return result;

            }

            static seeta::Image ImageFillBorder( const seeta::Image &img, const seeta::Rect &rect ) {
                int crop_x1 = std::max( 0, rect.x );
                int crop_y1 = std::max( 0, rect.y );
                int crop_x2 = std::min( img.width() - 1, rect.x + rect.width - 1 ); // Í¼Ïñ·¶Î§ 0µ½cols-1, 0µ½rows-1
                int crop_y2 = std::min( img.height() - 1, rect.y + rect.height - 1 );
                seeta::Image roi_img = seeta::crop( img, seeta::Region( crop_y1, crop_y2, crop_x1, crop_x2 ) ); // ×ó°üº¬£¬ÓÒ²»°üº¬

                int left_x = ( -rect.x );
                int top_y = ( -rect.y );
                int right_x = rect.x + rect.width - img.width();
                int down_y = rect.y + rect.height - img.height();
                left_x = ( left_x > 0 ? left_x : 0 );
                right_x = ( right_x > 0 ? right_x : 0 );
                top_y = ( top_y > 0 ? top_y : 0 );
                down_y = ( down_y > 0 ? down_y : 0 );
                auto imgFill = copyMakeBorder( roi_img, top_y, down_y, left_x, right_x );

                return imgFill;
            }

            static float ReBlur( const unsigned char *data, int width, int height ) {
                float blur_val = 0.0;
                float kernel[9] = { 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0 };
                float *BVer = new float[width * height];
                float *BHor = new float[width * height];

                float filter_data = 0.0;
                for( int i = 0; i < height; ++i ) { 
                    for( int j = 0; j < width; ++j ) {
                        if( i < 4 || i > height - 5 ) {

                            BVer[i * width + j] = data[i * width + j];
                        }
                        else {
                            filter_data = kernel[0] * data[( i - 4 ) * width + j] + kernel[1] * data[( i - 3 ) * width + j] + kernel[2] * data[( i - 2 ) * width + j] +
                                          kernel[3] * data[( i - 1 ) * width + j] + kernel[4] * data[( i ) * width + j] + kernel[5] * data[( i + 1 ) * width + j] +
                                          kernel[6] * data[( i + 2 ) * width + j] + kernel[7] * data[( i + 3 ) * width + j] + kernel[8] * data[( i + 4 ) * width + j];
                            BVer[i * width + j] = filter_data;
                        }

                        if( j < 4 || j > width - 5 ) {
                            BHor[i * width + j] = data[i * width + j];
                        }
                        else {
                            filter_data = kernel[0] * data[i * width + ( j - 4 )] + kernel[1] * data[i * width + ( j - 3 )] + kernel[2] * data[i * width + ( j - 2 )] +
                                          kernel[3] * data[i * width + ( j - 1 )] + kernel[4] * data[i * width + j] + kernel[5] * data[i * width + ( j + 1 )] +
                                          kernel[6] * data[i * width + ( j + 2 )] + kernel[7] * data[i * width + ( j + 3 )] + kernel[8] * data[i * width + ( j + 4 )];
                            BHor[i * width + j] = filter_data;
                        }

                    }
                }
                float D_Fver = 0.0;
                float D_FHor = 0.0;
                float D_BVer = 0.0;
                float D_BHor = 0.0;
                float s_FVer = 0.0;
                float s_FHor = 0.0;
                float s_Vver = 0.0;
                float s_VHor = 0.0;
                for( int i = 1; i < height; ++i ) {
                    for( int j = 1; j < width; ++j ) {
                        D_Fver = std::abs( ( float )data[i * width + j] - ( float )data[( i - 1 ) * width + j] );
                        s_FVer += D_Fver;
                        D_BVer = std::abs( ( float )BVer[i * width + j] - ( float )BVer[( i - 1 ) * width + j] );
                        s_Vver += std::max( ( float )0.0, D_Fver - D_BVer );

                        D_FHor = std::abs( ( float )data[i * width + j] - ( float )data[i * width + ( j - 1 )] );
                        s_FHor += D_FHor;
                        D_BHor = std::abs( ( float )BHor[i * width + j] - ( float )BHor[i * width + ( j - 1 )] );
                        s_VHor += std::max( ( float )0.0, D_FHor - D_BHor );
                    }
                }
                float b_FVer = ( s_FVer - s_Vver ) / s_FVer;
                float b_FHor = ( s_FHor - s_VHor ) / s_FHor;
                blur_val = std::max( b_FVer, b_FHor );

                delete[] BVer;
                delete[] BHor;

                return blur_val;
            }

            float ClarityEstimate( const SeetaImageData &image, const SeetaRect &info ) {
                if( !image.data || info.width < 9 || info.height < 9 ) return 0.0;
                seeta::Image color_data( image.data, image.width, image.height, image.channels );
                seeta::Image gray_data = seeta::gray( color_data );

                seeta::Image src_data = seeta::crop( gray_data, seeta::Rect( info.x, info.y, info.width, info.height ) );
                float blur_val = ReBlur( src_data.data(), src_data.width(), src_data.height() );
                float clarity = 1.0 - blur_val;

                float T1 = 0.3;
                float T2 = 0.55;
                if( clarity <= T1 ) {
                    clarity = 0.0;
                }
                else
                    if( clarity >= T2 ) {
                        clarity = 1.0;
                    }
                    else {
                        clarity = ( clarity - T1 ) / ( T2 - T1 );
                    }

                return clarity;
            }


            static bool CropFace( const seeta::Image &srcImg, const SeetaPointF *llpoint, seeta::Image &dstImg, SeetaPointF *finalPoint = nullptr ) {
                const int cropFaceSize = 512;

                if( dstImg.width() != cropFaceSize || dstImg.height() != cropFaceSize || dstImg.channels() != srcImg.channels() ) {
                    dstImg = seeta::Image( cropFaceSize, cropFaceSize, srcImg.channels() );
                }

                float mean_shape[10] =
                {
                    89.3095f + cropFaceSize / 4,
                    72.9025f + cropFaceSize / 4,
                    169.3095f + cropFaceSize / 4,
                    72.9025f + cropFaceSize / 4,
                    127.8949f + cropFaceSize / 4,
                    127.0441f + cropFaceSize / 4,
                    96.8796f + cropFaceSize / 4,
                    184.8907f + cropFaceSize / 4,
                    159.1065f + cropFaceSize / 4,
                    184.7601f + cropFaceSize / 4,
                };
                float points[10];
                for( int i = 0; i < 5; ++i ) {
                    points[2 * i] = float( llpoint[i].x );
                    points[2 * i + 1] = float( llpoint[i].y );
                }
                float tempPoint[10];

                face_crop_core_ex(
                    srcImg.data(), srcImg.width(), srcImg.height(), srcImg.channels(),
                    dstImg.data(),
                    cropFaceSize, cropFaceSize,
                    points, 5, mean_shape,
                    cropFaceSize, cropFaceSize,
                    0, 0, 0, 0,
                    tempPoint,
                    LINEAR,
                    NEAREST_PADDING );

                if( finalPoint != nullptr ) {
                    for( int i = 0; i < 5; ++i ) {
                        finalPoint[i].x = tempPoint[2 * i];
                        finalPoint[i].y = tempPoint[2 * i + 1];
                    }
                }

                return true;
            }

            bool vipl_BGR2YCrCb( int width, int height, int channels, const unsigned char *BGR_data, unsigned char *YCrCb_data ) {
                if( channels != 3 ) return false;
                for( int i = 0; i < width * height; ++i, BGR_data += 3, YCrCb_data += 3 ) {
                    unsigned char B = BGR_data[0];
                    unsigned char G = BGR_data[1];
                    unsigned char R = BGR_data[2];
                    unsigned int Y = ( B * 1868 + G * 9617 + R * 4899 + 8192 ) / 16384;
                    unsigned int U = ( ( B - Y ) * 9241 + 8192 ) / 16384 + 128;
                    unsigned int V = ( ( R - Y ) * 11682 + 8192 ) / 16384 + 128;
                    //      R = Y + 1.14 * V;
                    //      G = Y - 0.39 * U - 0.58 * V;
                    //      B = Y + 2.03 * U;
                    if( Y < 0 ) Y = 0;
                    YCrCb_data[0] = static_cast<unsigned char>( Y );
                    YCrCb_data[1] = static_cast<unsigned char>( V );
                    YCrCb_data[2] = static_cast<unsigned char>( U );
                }
                return true;
            }

            float score_face( const SeetaImageData &ViplImage, const SeetaRect &face, const SeetaPointF5 &points ) {
                seeta::Image frame( ViplImage.data, ViplImage.width, ViplImage.height, ViplImage.channels );
                seeta::Image crop;
                CropFace( frame, points, crop );

                vipl_BGR2YCrCb( crop.width(), crop.height(), crop.channels(), crop.data(), crop.data() );

                crop = seeta::resize( crop, seeta::Size( 256, 256 ) );

                auto tensor = tensor::build( UINT8, {1, 256, 256, 3}, crop.data() );
                m_bench.input( 0, tensor );
                m_bench.run();
                auto output = tensor::cast( FLOAT32, m_bench.output( 0 ) );

                return output.data<float>( 1 );

            }

            bool has_box( const SeetaImageData &image ) {
                seeta::Image frame( image.data, image.width, image.height, image.channels );

                bool has_box = false;
                if( m_box_detetor != nullptr ) {
                    auto boxs = m_box_detetor->Detect( frame );
                    has_box = boxs.size > 0;
                }
                return has_box;
            }


            FaceAntiSpoofing::Status predict( const SeetaImageData &ViplImage, const SeetaRect &face, const SeetaPointF5 &points ) {
                orz::ctx::lite::bind<orz::Shotgun> _bind_gun( m_gun.get() );

                this->clarity = ClarityEstimate( ViplImage, face );

                auto has_box = this->has_box( ViplImage );
                float result = has_box ? 0.0 : this->score_face( ViplImage, face, points );

                orz::Log( orz::DEBUG ) << "Resut: " << clarity << " " << result;

                this->passive_result = result;

                if( this->passive_result >= this->fuse_threshold ) {
                    if( this->clarity >= this->clarity_threshold ) {
                        return FaceAntiSpoofing::REAL;
                    }
                    else {
                        return FaceAntiSpoofing::FUZZY;
                    }
                }
                else {
                    return FaceAntiSpoofing::SPOOF;
                }
            }


            FaceAntiSpoofing::Status predictVideo( const SeetaImageData &ViplImage, const SeetaRect &face, const SeetaPointF5 &points ) {
                orz::ctx::lite::bind<orz::Shotgun> _bind_gun( m_gun.get() );

                this->clarity = ClarityEstimate( ViplImage, face );

                auto has_box = this->has_box( ViplImage );
                float result = has_box ? 0.0 : this->score_face( ViplImage, face, points );

                this->passive_result = result;

                if( frameNumQueue > fasVideoFrameNumThreshold ) {

                    fasVideoDeque.pop_front();
                }
                fasVideoDeque.push_back( result );
                orz::Log( orz::DEBUG ) << "Resut: " << clarity << " " << result;

                frameNumQueue++;

                if( fasVideoDeque.size() < fasVideoFrameNumThreshold ) {
                    return FaceAntiSpoofing::DETECTING;
                }

                else {
                    std::deque<double>::iterator ator;
                    double passiveResultVideoAll = 0.0;
                    for( ator = fasVideoDeque.begin(); ator != fasVideoDeque.end() ; ator++ ) {
                        passiveResultVideoAll += *ator;
                    }

                    passiveResultVideoMean = passiveResultVideoAll / fasVideoFrameNumThreshold;

                    if( passiveResultVideoMean >= this->fuse_threshold ) {
                        return FaceAntiSpoofing::REAL;
                    }
                    else {
                        return FaceAntiSpoofing::SPOOF;
                    }
                }


            }

            void SetThreshold( double value1, double value2 ) {
                clarity_threshold = value1;
                fuse_threshold = value2;
            }

			void SetBoxThresh(float thresh)
			{
				m_box_detetor->SetThreshold(thresh);
			}
			
			float GetBoxThresh()const
			{
				return m_box_detetor->GetThreshold();
			}
			
            std::string GetLog() const {
                std::string log( " clarity = " + std::to_string( clarity ) + " passive result = " + std::to_string( this->passive_result ) );
                return log;
            }
            void resetVideo() {
                while( !fasVideoDeque.empty() ) {
                    fasVideoDeque.pop_back();
                }
                passiveResultVideoMean = 0.0;
                frameNumQueue = 0;
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

            void set(FaceAntiSpoofing::Property property, double value) {
                switch (property) {
                    default:
                        break;
  
                    case FaceAntiSpoofing::PROPERTY_NUMBER_THREADS:
                    {
                        if (value < 1) value = 1;
                        auto threads = int(value);
                        m_number_threads = threads;
                        m_bench.set_computing_thread_number(threads);
                        break;
                    }
                    case FaceAntiSpoofing::PROPERTY_ARM_CPU_MODE:
                        set_cpu_affinity(int32_t(value));
                        break;
                }
            }
            double get(FaceAntiSpoofing::Property property) const {
                switch (property) {
                    default:
                        return 0;
 
                    case FaceAntiSpoofing::PROPERTY_NUMBER_THREADS:
                        return m_number_threads;
                    case FaceAntiSpoofing::PROPERTY_ARM_CPU_MODE:
                        return get_cpu_affinity();
                }
            }


        public:
            std::shared_ptr<seeta::BoxDetector> m_box_detetor;

            ModelParam m_param;
            mutable Workbench m_bench;


            double clarity = 0;
            double passive_result = 0;

            double clarity_threshold = 0.3;
            double fuse_threshold = 0.80;
            std::deque<double> fasVideoDeque;

            int frameNumQueue = 0;  
            int fasVideoFrameNumThreshold = 10;
            double passiveResultVideoMean = 0.0; 

            std::shared_ptr<orz::Shotgun> m_gun;

            
            int32_t m_number_threads = 4;
            int m_cpu_affinity = -1;

        };

        class FaceAntiSpoofing::Implement {
        public:
            using Inner = FaceAntiSpoofingX;

            Implement( const seeta::ModelSetting &setting ) {
                std::vector<std::string> models;
                int index = 0;
                while (setting.model[index]) models.push_back(setting.model[index++]);
                if (models.empty())
                {
                    orz::Log(orz::FATAL) << LOG_HEAD << "Must input 1 or 2 models." << orz::crash;
                }


                seeta::ModelSetting anti_setting(setting);
                anti_setting.clear();
                anti_setting.append(models[0]);
                m_impl.reset( new Inner( anti_setting ) );

                if(models.size() > 1)
                {
                    seeta::ModelSetting boxs_setting(setting);
                    boxs_setting.clear();
                    boxs_setting.append(models[1]);
                    m_impl->load_box_model( boxs_setting );
                }
            }

            std::shared_ptr<Inner> m_impl;
        };

        FaceAntiSpoofing::FaceAntiSpoofing( const seeta::ModelSetting &setting )
            : m_impl( new Implement( setting ) )
        {

        }

        FaceAntiSpoofing::~FaceAntiSpoofing()
        {
            delete this->m_impl;
        }

        FaceAntiSpoofing::Status FaceAntiSpoofing::Predict( const SeetaImageData &image, const SeetaRect &face,
                const SeetaPointF *points ) const
        {

            SeetaPointF vpoints[5];
            for( int i = 0; i < 5; ++i )
            {
                vpoints[i].x = points[i].x;
                vpoints[i].y = points[i].y;
            }
            auto vstatus = m_impl->m_impl->predict( image, face, vpoints );
            return Status( vstatus );
        }

        FaceAntiSpoofing::Status FaceAntiSpoofing::PredictVideo( const SeetaImageData &image, const SeetaRect &face,
                const SeetaPointF *points ) const
        {

            SeetaPointF vpoints[5];
            for( int i = 0; i < 5; ++i )
            {
                vpoints[i].x = points[i].x;
                vpoints[i].y = points[i].y;
            }
            auto vstatus = m_impl->m_impl->predictVideo( image, face, vpoints );
            return Status( vstatus );
        }

        void FaceAntiSpoofing::ResetVideo()
        {
            m_impl->m_impl->resetVideo();
        }

        void FaceAntiSpoofing::GetPreFrameScore( float *clarity, float *reality )
        {
            if( clarity )
            {
                *clarity = float( m_impl->m_impl->clarity );
            }
            if( reality )
            {
                *reality = float( m_impl->m_impl->passive_result );
            }
        }

        void FaceAntiSpoofing::SetVideoFrameCount( int32_t number )
        {
            m_impl->m_impl->fasVideoFrameNumThreshold = number > 1 ? number : 1;
        }

        int32_t FaceAntiSpoofing::GetVideoFrameCount() const
        {
            return m_impl->m_impl->fasVideoFrameNumThreshold;
        }

        void FaceAntiSpoofing::SetThreshold( float clarity, float reality )
        {
            m_impl->m_impl->SetThreshold( clarity, reality );
        }

        void FaceAntiSpoofing::GetThreshold( float *clarity, float *reality ) const
        {
            if( clarity )
            {
                *clarity = float( m_impl->m_impl->clarity_threshold );
            }
            if( reality )
            {
                *reality = float( m_impl->m_impl->fuse_threshold );
            }
        }

        void FaceAntiSpoofing::set(FaceAntiSpoofing::Property property, double value) {
            m_impl->m_impl->set(property, value);
        }

        double FaceAntiSpoofing::get(FaceAntiSpoofing::Property property) const {
            return m_impl->m_impl->get(property);
        }
		
		void FaceAntiSpoofing::SetBoxThresh(float thresh)
		{
			m_impl->m_impl->SetBoxThresh(thresh);
		}
		
		float FaceAntiSpoofing::GetBoxThresh()const
		{
			return m_impl->m_impl->GetBoxThresh();
		}

    }
}
