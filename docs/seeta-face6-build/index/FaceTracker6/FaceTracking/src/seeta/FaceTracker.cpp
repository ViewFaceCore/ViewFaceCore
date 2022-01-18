#include "seeta/FaceTracker.h"
#include <seeta/FaceDetector.h>
#include "seeta/CTrackingFaceInfo.h"

#include <orz/utils/log.h>

#include <iostream>
#include <mutex>
#include <array>
#include <cmath>
#include <fstream>
#include <sstream>


#include <time.h>
#include <cstring>
#include <vector>
#include <algorithm>

using namespace std;

template <typename A, typename B>
static inline A max( const A &a, const B &b )
{
    return a > b ? a : static_cast<A>( b );
}

template <typename A, typename B>
static inline A min( const A &a, const B &b )
{
    return a < b ? a : static_cast<A>( b );
}


const int G_MAX_MISSING_STEP = 3;


namespace seeta
{
    namespace v6
    {
        class FaceTracker::Implement {
        public:
            using self = Implement;

            Implement( const SeetaModelSetting &setting, int videowidth, int videoheight ) {
                
                setParameter( 10, videowidth, videoheight, 40, 40, 0, setting);
                _redetect_flag = 1;
                _count = 0;
                _detect_flag = 1;
                _person_id = 0;
                _track_count = 0;
                _max_intersect_radio = 0.3f;

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

            SeetaTrackingFaceInfoArray Track( const SeetaImageData &image, int frame_no = -1 ) {
                if( image.width != video_width ||
                        image.height != video_height ) {
                    orz::Log( orz::FATAL ) << "Track " <<
                                           "Input unsupported image size (" << image.width << ", " << image.height << ") vs. (" <<
                                           video_width << ", " << video_height << ") expected." << orz::crash;
                }
                if( image.channels != 3 ) {
                    orz::Log( orz::FATAL ) << "Track " <<
                                           "Input unsupported image channels (" << image.channels << ") vs. (3) expected." << orz::crash;
                }

                if( frame_no >= 0 ) {
                    m_frame_no = frame_no;
                }

                inputBGR24( image.data, m_frame_no++ );
                auto infos = getFaceInfo();

                std::vector<SeetaTrackingFaceInfo> now_faces;
                now_faces.reserve( infos.size() );
                for( auto &vface : infos ) {
                    if( vface.step > 0 ) continue;

                    now_faces.push_back(vface);
                }

                if( m_video_stable ) {
                    for( auto &now_face : now_faces ) {
                        for( auto &pre_face : m_pre_faces ) {
                            if( now_face.PID != pre_face.PID ) continue;
                            if( IoU( now_face.pos, pre_face.pos ) > 0.85 )
                                now_face = pre_face;
                            else
                                if( IoU( now_face.pos, pre_face.pos ) > 0.6 ) {
                                    now_face.pos.x = int( std::lround( float( now_face.pos.x + pre_face.pos.x ) / 2.0f ) );
                                    now_face.pos.y = int( std::lround( float( now_face.pos.y + pre_face.pos.y ) / 2.0f ) );
                                    now_face.pos.width = int( std::lround( float( now_face.pos.width + pre_face.pos.width ) / 2.0f ) );
                                    now_face.pos.height = int( std::lround( float( now_face.pos.height + pre_face.pos.height ) / 2.0f ) );
                                }
                        }
                    }
                }

                //uniq_face_vec( now_faces, _max_intersect_radio, true);
                m_pre_faces = now_faces;

                const SeetaTrackingFaceInfoArray result = { m_pre_faces.data(), int( m_pre_faces.size() ) };
                return result;
                
            }


            void setvideosize(int vid_width, int vid_height)
            {
                video_width = vid_width;    //video width
                video_height = vid_height;  //video height
                _cur_rgb_img.reset( new unsigned char[video_height * video_width * 3], std::default_delete<unsigned char[]>() );
                _local_iamge_buffer_size = video_height * video_width * 3;
                _local_image_buffer.reset( new unsigned char[_local_iamge_buffer_size], std::default_delete<unsigned char[]>() );
            }

            void setParameter( int interval, int vid_width, int vid_height, int min_face_detect_size, int key_frame_size, float threshold, const seeta::ModelSetting &setting) {

                track_interval = interval;  //frame interval
                video_width = vid_width;    //video width
                video_height = vid_height;  //video height

                _min_face_size = min_face_detect_size;

                _image_pyramid_scale_factor = 1.414f;

                _face_detector.reset( new seeta::FaceDetector( setting ) );
                _face_detector->set( seeta::FaceDetector::Property::PROPERTY_MIN_FACE_SIZE, _min_face_size );

                _cur_rgb_img.reset( new unsigned char[video_height * video_width * 3], std::default_delete<unsigned char[]>() );
                _local_iamge_buffer_size = video_height * video_width * 3;
                _local_image_buffer.reset( new unsigned char[_local_iamge_buffer_size], std::default_delete<unsigned char[]>() );
            }

            int redetect_track_rst( std::vector<SeetaTrackingFaceInfo> &redetect_track_vec, uint8_t *bgr24_data, int width, int height, float expand_time, int flag, bool strict=false ) {
                float expand_time_w = 0.0f;
                float expand_time_h = 0.0f;
                int ori_vec_size = ( int )redetect_track_vec.size();

                for( int i = 0; i < ori_vec_size; ++i ) {
                    SeetaRect cur_rect = redetect_track_vec[i].pos;
                    float radio = ( float )cur_rect.height / ( float )cur_rect.width;
                    if( radio > expand_time ) {
                        expand_time_w = ( expand_time - 1.0f ) / 2.0f;
                        expand_time_h = 0.0f;
                    }
                    else
                        if( radio > 1.15 ) {
                            expand_time_w = ( expand_time - 1.0f ) / 2.0f;
                            expand_time_h = ( expand_time / radio * 1.05f - 1.0f ) / 2.0f;
                        }
                        else
                            if( radio > 0.87 ) {
                                expand_time_w = ( expand_time - 1.0f ) / 2.0f;
                                expand_time_h = expand_time_w;
                            }
                            else
                                if( radio > ( 1.0f / expand_time ) ) {
                                    expand_time_w = ( expand_time * radio * 1.05f - 1.0f ) / 2.0f;
                                    expand_time_h = ( expand_time - 1.0f ) / 2.0f;
                                }
                                else {
                                    expand_time_w = 0.0f;
                                    expand_time_h = ( expand_time - 1.0f ) / 2.0f;
                                }
                    int half_expand_width = ( int )( expand_time_w * cur_rect.width );
                    int half_expand_height = ( int )( expand_time_h * cur_rect.height );
                    int new_x = max( cur_rect.x - half_expand_width, 0 );
                    int new_y = max( cur_rect.y - half_expand_height, 0 );
                    int new_width = min( cur_rect.x + cur_rect.width + half_expand_width, width ) - new_x;
                    int new_height = min( cur_rect.y + cur_rect.height + half_expand_height, height ) - new_y;

                    uint8_t *src_data = bgr24_data + ( new_y * width + new_x ) * 3;

                    // ±£Ö¤»º³å´óÐ¡×ã¹»
                    if( size_t( new_width * new_height * 3 ) > _local_iamge_buffer_size ) {
                        _local_iamge_buffer_size = new_width * new_height * 3;
                        _local_image_buffer.reset( new unsigned char[_local_iamge_buffer_size], std::default_delete<unsigned char[]>() );
                    }
                    uint8_t *dst_data = _local_image_buffer.get();

                    SeetaImageData image_data;
                    image_data.width = new_width;
                    image_data.height = new_height;
                    image_data.channels = 3;
                    image_data.data = dst_data;

                    for( int h = 0; h < new_height; ++h ) {
                        memcpy( dst_data, src_data, new_width * 3 );
                        dst_data += new_width * 3;
                        src_data += width * 3;
                    }

                    float fscale = min<float>(cur_rect.width / video_width, cur_rect.height / video_height);
                    int local_min_face = max<int>(20, min<int>(_min_face_size, _min_face_size * fscale * 1.4));
               		
                    _face_detector->set( seeta::FaceDetector::Property::PROPERTY_MIN_FACE_SIZE, local_min_face);
                    std::vector<SeetaRect> faces;
                    auto face_info_array = _face_detector->detect( image_data );
                    for( int i = 0; i < face_info_array.size; ++i ) {
                        SeetaRect seeta_rect = face_info_array.data[i].pos;
                        faces.push_back( seeta_rect );
                    }

                    size_t face_num = faces.size();

                    if( face_num <= 0 ) {
                        if( !strict && redetect_track_vec[i].step < G_MAX_MISSING_STEP ) {
                            redetect_track_vec[i].step++;
                            
                            SeetaTrackingFaceInfo face_info_node = redetect_track_vec[i];
                            face_info_node.score = 0;
                            redetect_track_vec.push_back( face_info_node );
                        }
                        continue;
                    }
                    for( int j = 0; j < 1; ++j ) { // only redectect local face
                        SeetaRect new_face_info;
                        new_face_info.x = new_x + faces[j].x;
                        new_face_info.y = new_y + faces[j].y;
                        new_face_info.width = faces[j].width;
                        new_face_info.height = faces[j].height;
                        SeetaTrackingFaceInfo face_info_node;
                        face_info_node.PID = redetect_track_vec[i].PID;
                        face_info_node.frame_no = redetect_track_vec[i].frame_no;
                        face_info_node.score = 0;
                        face_info_node.pos = new_face_info;
                        face_info_node.step = 0;
                        redetect_track_vec.push_back( face_info_node );

                    }
                }

                std::vector<SeetaTrackingFaceInfo>::iterator bit = redetect_track_vec.begin();
                std::vector<SeetaTrackingFaceInfo>::iterator eit = redetect_track_vec.begin();
                while( ori_vec_size > 0 ) {
                    ++eit;
                    --ori_vec_size;
                }
                redetect_track_vec.erase( bit, eit ); //¿ÉÓÅ»¯ !!!!!

                return 0;
            }

            void uniq_face_vec( std::vector<SeetaTrackingFaceInfo> &redetect_track_vec, float max_intersect_radio, bool matchscore = false ) {
                size_t vec_size = redetect_track_vec.size();
                ( void )( vec_size );
                std::vector<SeetaTrackingFaceInfo>::iterator cur_it;
                std::vector<SeetaTrackingFaceInfo>::iterator sli_it;
                for( cur_it = redetect_track_vec.begin(); cur_it != redetect_track_vec.end(); ++cur_it ) {
                    SeetaRect &cur_face = ( *cur_it ).pos;
                    int cur_area = ( int )( cur_face.width * cur_face.height * max_intersect_radio );

                    sli_it = cur_it + 1;
                    while( sli_it != redetect_track_vec.end() ) {
                        SeetaRect &sli_face = ( *sli_it ).pos;
                        int x_begin = max( cur_face.x, sli_face.x );
                        int x_end = min( cur_face.x + cur_face.width, sli_face.x + sli_face.width );
                        int y_begin = max( cur_face.y, sli_face.y );
                        int y_end = min( cur_face.y + cur_face.height, sli_face.y + sli_face.height );

                        if( x_begin >= x_end || y_begin >= y_end ) {
                            ++sli_it;
                            continue;
                        }
                        int intersect_area = ( x_end - x_begin ) * ( y_end - y_begin );
                        int sli_area = ( int )( sli_face.width * sli_face.height * max_intersect_radio );
                        if( intersect_area > sli_area || intersect_area > cur_area ) {
                            if(matchscore)
                            {
                                if(cur_it->PID > sli_it->PID)
                                {
                                    sli_it = redetect_track_vec.erase( sli_it ); //Ïà½»ÇøÓò³¬¹ýMAX_INTERSECT_RADIO, É¾³ý¶ÔÓ¦½Úµã
                                }else
                                {
                                    if(cur_it->score >= sli_it->score)
                                    {
                                        sli_it = redetect_track_vec.erase( sli_it ); //Ïà½»ÇøÓò³¬¹ýMAX_INTERSECT_RADIO, É¾³ý¶ÔÓ¦½Úµã
                                    }else
                                    {
                                        cur_it = redetect_track_vec.erase( cur_it );
                                        --cur_it;
                                    }
                                }
                            }else
                            {
                                if( sli_face.width > cur_face.width && sli_face.height > cur_face.height ) {
                                    cur_it = redetect_track_vec.erase( cur_it );
                                    --cur_it;
                                    break;
                                }
                                else {
                                    sli_it = redetect_track_vec.erase( sli_it ); //Ïà½»ÇøÓò³¬¹ýMAX_INTERSECT_RADIO, É¾³ý¶ÔÓ¦½Úµã
                                }
                             }
                        }
                        else {
                            ++sli_it;
                        }
                    }  //while (sli_it != _redetect_track_vec.end())
                }

            }

            int inputBGR24( unsigned char *bgr24, int frameID ) {
                memcpy( _cur_rgb_img.get(), bgr24, video_width * video_height * 3 );

                _frame_id = frameID;

                //¼ì²âÈËÁ³
                if( _detect_flag == 1 && _redetect_flag == 1 ) {
                    SeetaImageData image_data_rgb;
                    image_data_rgb.width = video_width;
                    image_data_rgb.height = video_height;
                    image_data_rgb.channels = 3;
                    image_data_rgb.data = _cur_rgb_img.get();

                    _face_detector->set( seeta::FaceDetector::Property::PROPERTY_MIN_FACE_SIZE, _min_face_size );

                    std::vector<SeetaRect> cur_faces;
                    auto face_info_array = _face_detector->detect( image_data_rgb );
                    for( int i = 0; i < face_info_array.size; ++i ) {
                        SeetaRect seeta_rect = face_info_array.data[i].pos;
                        cur_faces.push_back( seeta_rect );
                    }

                    size_t cur_face_num = cur_faces.size();

                    //std::cout<< "----cur face num:" << cur_face_num << std::endl;
                    if( cur_face_num == 0 ) {
                        _redetect_flag = 0;
                        _count++;
                    }
                    size_t cur_vec_size = _cur_track_vec.size();
                    ( void )( cur_vec_size );


                    size_t tracingNum = _tracing_vec.size()/*cur_vec_size*/;

                    _cur_track_vec.clear();

                    std::vector<int> matchResult( tracingNum );
                    std::vector<bool> newFaceFlag( cur_face_num );
                    for( size_t j = 0; j < cur_face_num; j++ ) {
                        newFaceFlag[j] = true;  //±êÖ¾³õÊ¼»¯
                    }
                    for( size_t i = 0; i < tracingNum; i++ ) {
                        matchResult[i] = -1;
                        double max_overlap = 0;

                        for( size_t j = 0; j < cur_face_num; j++ ) {

                            int minx = max( _tracing_vec[i].pos.x, cur_faces[j].x );
                            int miny = max( _tracing_vec[i].pos.y, cur_faces[j].y );
                            int maxx = min( _tracing_vec[i].pos.x + _tracing_vec[i].pos.width, cur_faces[j].x + cur_faces[j].width );
                            int maxy = min( _tracing_vec[i].pos.y + _tracing_vec[i].pos.height, cur_faces[j].y + cur_faces[j].height );
                            if( ( minx > maxx ) || ( miny > maxy ) ) {
                                continue;
                            }
                            double area1 = _tracing_vec[i].pos.width * _tracing_vec[i].pos.height;
                            double area2 = cur_faces[j].width * cur_faces[j].height;
                            double total_area = area1 + area2;
                            double intersection = ( maxx - minx ) * ( maxy - miny );
                            double overlap = intersection / ( total_area - intersection );
                            if( overlap > max_overlap ) {
                                max_overlap = overlap;
                                matchResult[i] = int( j );
                            }

                        }
                        if( max_overlap < 0.1 ) { //0.25
                            matchResult[i] = -1;
                        }
                        if( matchResult[i] != -1 ) {
                            newFaceFlag[matchResult[i]] = false;
                        }
                    }
                    for( size_t j = 0; j < cur_face_num; j++ ) {
                        if( ( newFaceFlag[j] == true ) /* && (realFaces[j] == true) */ ) { //ÐÂÔöÄ¿±ê
                            SeetaTrackingFaceInfo tmpResult;
                            tmpResult.PID =  _person_id++;
                            tmpResult.frame_no = _frame_id;
                            tmpResult.pos.x = cur_faces[j].x;
                            tmpResult.pos.y = cur_faces[j].y;
                            tmpResult.pos.width = cur_faces[j].width;
                            tmpResult.pos.height = cur_faces[j].height;
                            tmpResult.score = 0;
                            tmpResult.step = 0;

                            _cur_track_vec.push_back( tmpResult ); //¼ì²âµ½µÄ£¬Ö±½Ó½ø


                        }
                    }

                    _redetect_track_vec.clear();
                    for( size_t i = 0; i < tracingNum; i++ ) {
                        if( matchResult[i] == -1 ) {
                            bool tmpTrackFlag[1];
                            tmpTrackFlag[0] = true;

                            if( tmpTrackFlag[0] ) {
                                _redetect_track_vec.push_back( _tracing_vec[i] ); //¸ú×Ùµ½µÄ½á¹û,´æ´¢ÆðÀ´,ºóÐøÖØÐÂ½øÐÐÈËÁ³¼ì²â
                            }

                            if( _redetect_track_vec.size() > 0 ) {

                                uniq_face_vec( _redetect_track_vec, _max_intersect_radio ); //½ÚµãÈ¥ÖØ£¬·ÀÖ¹ÉÏÒ»Ö¡²»Í¬idµÄÈËÔÚµ±Ç°Ö¡´íÎóµØ¸ú×ÙÎªÍ¬Ò»¸öÈË  // X?

                                redetect_track_rst(_redetect_track_vec, _cur_rgb_img.get(), video_width, video_height, 2.0f, true );  // ÑÏ¸ñÄ£Ê½£¬¼ì²â²»µ½ÈËÁ³²»Óè¸ú×Ù
                                size_t redet_size = _redetect_track_vec.size();

                                for( size_t j = 0; j < redet_size; ++j ) {
                                    _cur_track_vec.push_back( _redetect_track_vec[j] ); //¼ì²âµ½µÄ£¬Ö±½Ó½ø
                                }
                            }
                        }
                        else {  //Î´ÏûÊ§Ä¿±ê

                            // VIPLFaceInfo tmpBBox = cur_faces[matchResult[i]];
                            SeetaRect tmpBBox = cur_faces[matchResult[i]];
                            SeetaTrackingFaceInfo tmpResult;
                            tmpResult.PID =  _tracing_vec[i].PID;
                            tmpResult.frame_no = _frame_id;
                            tmpResult.pos.x = tmpBBox.x;
                            tmpResult.pos.y = tmpBBox.y;
                            tmpResult.pos.width = tmpBBox.width;
                            tmpResult.pos.height = tmpBBox.height;
                            tmpResult.score = 0;
                            tmpResult.step = 0;
                            _cur_track_vec.push_back( tmpResult ); //¼ì²âµ½µÄ£¬Ö±½Ó½ø

                        }
                    }

                    if( _cur_track_vec.size() != 0 ) {
                        _track_flag = 1;
                        _detect_flag = 0;
                    }
                }
                else {     //×·×Ù
                    if( _track_flag == 1 ) {
                        _redetect_track_vec.clear();
                        //³õÊ¼»¯

                        int nCurrentFaceNum = ( int )_cur_track_vec.size();
                        ( void )( nCurrentFaceNum );

                        redetect_track_rst( /*cur_rgb_img,*/ _cur_track_vec, _cur_rgb_img.get(), video_width, video_height, 2.0f, false );  // ·ÇÑÏ¸ñÄ£Ê½£¬ÔÊÐí¾Ö²¿¼ì²âÂ©¼ì

                        _track_count++;

                        //10Ö¡Ò»ÖÜÆÚ
                        if( _track_count == track_interval ) {
                            _detect_flag = 1;
                            _track_flag = 0;
                            _track_count = 0;

                        }
                    }
                    else {
                        _count++;

                        if( _count == 5 ) {
                            _count = 0;
                            _redetect_flag = 1;
                        }
                    }
                }

                for( vector<SeetaTrackingFaceInfo>::iterator iter = _tracing_vec.begin(); iter != _tracing_vec.end(); ) {
                    if( ( *iter ).step >= G_MAX_MISSING_STEP ) {
                        iter = _tracing_vec.erase( iter );
                        continue;
                    }
                    ++iter->step;
                    ++iter;
                }


                uniq_face_vec( _cur_track_vec, _max_intersect_radio, true);
                for( vector<SeetaTrackingFaceInfo>::iterator iter_cur = _cur_track_vec.begin(); iter_cur != _cur_track_vec.end(); iter_cur++ ) {
                    for( vector<SeetaTrackingFaceInfo>::iterator iter = _tracing_vec.begin(); iter != _tracing_vec.end(); ) {
                        if( ( *iter ).PID == ( *iter_cur ).PID ) {
                            iter = _tracing_vec.erase( iter );
                            continue;
                        }
                        ++iter;
                    }
                }


               
                _tracing_vec.insert( _tracing_vec.end(), _cur_track_vec.begin(), _cur_track_vec.end() );

                uniq_face_vec( _tracing_vec, _max_intersect_radio, true);
                // no need to copy _pre
                // memcpy(_pre_rgb_img.get(), _cur_rgb_img.get(), video_width * video_height * 3 * sizeof(unsigned char));  //_pre_rgb_img;

                return 1;
            }


            std::vector<SeetaTrackingFaceInfo> getFaceInfo() {
                auto result = _cur_track_vec;
                result.erase( std::remove_if( result.begin(), result.end(), []( SeetaTrackingFaceInfo info ) {
                    return info.step > 0;
                } ), result.end() );
                return result;
            }

            void Reset()
            {
                _redetect_flag = 1;
                _count = 0;
                _detect_flag = 1;
                _track_count = 0;

                _cur_track_vec.clear();
                _tracing_vec.clear();
                _redetect_track_vec.clear();
            }

            //////////////////////////////////////////////////////////
            bool m_video_stable = true;
            mutable std::vector<SeetaTrackingFaceInfo> m_pre_faces;
            mutable int m_frame_no = 0;

            ////////////////////////////////////////
            int video_width;
            int video_height;
            int track_interval;


            std::shared_ptr<seeta::FaceDetector> _face_detector;

            std::shared_ptr<unsigned char> _cur_rgb_img;
            std::shared_ptr<unsigned char> _local_image_buffer;
            size_t _local_iamge_buffer_size;

            std::vector<SeetaTrackingFaceInfo> _cur_track_vec;
            std::vector<SeetaTrackingFaceInfo> _tracing_vec;
            std::vector<SeetaTrackingFaceInfo> _redetect_track_vec;

            float _max_intersect_radio;
            int _frame_id;
            int _detect_flag;
            int _person_id;
            int _track_count;
            int _track_flag;
            int _redetect_flag;
            int _count;

            int32_t _min_face_size;
            int32_t _max_face_size;
            float _image_pyramid_scale_factor;

        };


        ///////////////////////////////////////////////////////
        FaceTracker::FaceTracker( const seeta::ModelSetting &setting, int video_width, int video_height )
            : m_impl( new Implement( setting, video_width, video_height ) )
        {
        }

        FaceTracker::~FaceTracker()
        {
            delete m_impl;
        }


        void FaceTracker::SetSingleCalculationThreads( int num )
        {
            auto *detector = m_impl->_face_detector.get();
            detector->set( seeta::FaceDetector::Property::PROPERTY_NUMBER_THREADS, num );
        }

        SeetaTrackingFaceInfoArray FaceTracker::Track( const SeetaImageData &image ) const
        {
            return m_impl->Track( image );
        }

        SeetaTrackingFaceInfoArray FaceTracker::Track( const SeetaImageData &image, int frame_no ) const
        {
            return m_impl->Track( image, frame_no );
        }

        void FaceTracker::SetMinFaceSize( int32_t size )
        {
            auto *detector = m_impl->_face_detector.get();
            detector->set( seeta::FaceDetector::Property::PROPERTY_MIN_FACE_SIZE, size );
            m_impl->_min_face_size = size; 
        }

        int32_t FaceTracker::GetMinFaceSize() const
        {
            auto *detector = m_impl->_face_detector.get();
            return detector->get( seeta::FaceDetector::Property::PROPERTY_MIN_FACE_SIZE );
        }

        void FaceTracker::SetThreshold( float thresh )
        {
            double value = thresh;
            auto *detector = m_impl->_face_detector.get();
            detector->set( seeta::FaceDetector::Property::PROPERTY_THRESHOLD, value );
        }

        float FaceTracker::GetThreshold() const
        {
            auto *detector = m_impl->_face_detector.get();
            double value = detector->get( seeta::FaceDetector::Property::PROPERTY_THRESHOLD);
            return (float)value;
        }

        void FaceTracker::SetVideoStable(bool stable)
        {
            m_impl->m_video_stable = stable;
        }

        bool FaceTracker::GetVideoStable( ) const
        {
            return m_impl->m_video_stable;
        }

        void FaceTracker::SetVideoSize(int vidwidth, int vidheight)
        {
            m_impl->setvideosize(vidwidth, vidheight);
        }
        
        void FaceTracker::SetInterval(int interval)
        {
            m_impl->track_interval = interval;
        }

        void FaceTracker::Reset()
        {
            m_impl->Reset();
        }
    }

}
