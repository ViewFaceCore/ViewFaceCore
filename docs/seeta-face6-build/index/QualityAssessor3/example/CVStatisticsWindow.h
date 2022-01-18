#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

class CVStatisticsWindow {
public:
    explicit CVStatisticsWindow( int max = 100, float font_scale = 0.5 ) : m_max( max ), m_font_scale( font_scale ),m_color(0,0,0) {
        m_mask = "MASK";
        m_lbn = "QUALITYOFLBN";
        m_light_value = -1;
        m_blur_value = -1;
        m_noise_value = -1;
        m_maskvalues.resize(5, -1);
              
    }

    void add_name( const std::string &name, int val = 0 ) {
        m_names.push_back( name );
        m_values.push_back( val );
    }

    void set_color(cv::Scalar color)
    {
        m_color = color;
    }
    void clear_name() {
        m_names.clear(), m_values.clear();
    }

    void set_name( int index, const std::string &name ) {
        m_names[index] = name;
    }

    size_t size() const {
        return m_names.size();
    }

    int &value( int index ) {
        return m_values[index];
    }
    const int &value( int index ) const {
        return m_values[index];
    }

    std::string &name( int index ) {
        return m_names[index];
    }
    const std::string &name( int index ) const {
        return m_names[index];
    }

    int &max() {
        return m_max;
    }
    const int &max() const {
        return m_max;
    }

    void value( int index, int val ) {
        m_values[index] = val;
    }

    void set_value( const std::string &name, int value ) {
        for(int i=0; i<m_names.size(); i++)
        {
            if(m_names[i] == name)
            {
                m_values[i] = value;
                break;
            }
        }
    }

    void set_name( const std::vector<std::string> &names ) {
        m_names = names;
        m_values.resize( names.size(), 0 );
    }
    void set_name( const std::vector<std::string> &names, const std::vector<int> &values ) {
        m_names = names;
        m_values = values;
    }

    void set_mask_name(const std::string & name) {
        m_mask = name;
    }
    void set_mask_values(const std::vector<int> &value) {
        m_maskvalues = value;
    }

    void set_lbn_name(const std::string & name) {
        m_lbn = name;
    }
    void set_light_value(int value) {
        m_light_value = value;
    }

    void set_blur_value(int value) {
        m_blur_value = value;
    }
    void set_noise_value(int value) {
        m_noise_value = value;
    }



    void imshow( const std::string &winname ) const {
        if( m_canvas.empty() ) return;
        cv::imshow( winname, m_canvas );
    }
    void imwrite( const std::string &filename ) const {
        if( m_canvas.empty() ) return;
        cv::imwrite( filename, m_canvas );
    }

    void fillRectangle( cv::Mat &img, const cv::Rect &rect, const cv::Scalar &scalar, int lineType = 8 /*cv::LINE_8*/, int shift = 0, cv::Point offset = cv::Point() ) {
        std::vector<cv::Point> points =
        {
            { rect.x, rect.y },
            { rect.x, rect.y + rect.height },
            { rect.x + rect.width, rect.y + rect.height },
            { rect.x + rect.width, rect.y },
        };
        const cv::Point *pts[1] = { points.data() };
        int npts[] = { points.size() };
        cv::fillPoly( img, pts, npts, 1, scalar, lineType, shift, offset );
    }

    void update() {
        if( m_names.empty() ) {
            m_canvas = cv::Mat();
            return;
        }
        size_t max_string_length = 0;
        for( auto &name : m_names ) if( name.length() > max_string_length ) max_string_length = name.length();

        if(max_string_length < m_mask.length())
        {
            max_string_length = m_mask.length();
        }

        if(max_string_length < m_lbn.length())
        {
            max_string_length = m_lbn.length();
        }
        int letter_width = 20 * m_font_scale;
        int letter_height = 20 * m_font_scale;

        int name_area_width = letter_width * ( max_string_length + 2 );
        int process_area_width = letter_width * 68;
        int percentage_area_width = letter_width * 10;

        int fieldwidth = letter_width * 6;

        int box_width = name_area_width + process_area_width + percentage_area_width;
        int box_height = letter_height * 3;

        int canvas_width = box_width;
        int canvas_height = box_height * (m_names.size() + 6) + 10;

        //std::cout << "------read 00----" << std::endl;
        //cv::Mat rightmat = cv::imread("/wqy/Downloads/right.jpg");

        //std::cout << "------read 11----" << std::endl;
        //cv::Mat tmp(40,40, rightmat.type());
        //cv::resize(rightmat, tmp, cv::Size(20,20), (0,0),(0,0));
        //std::cout << "------read ok----" << std::endl;
        m_canvas = cv::Mat( canvas_height, canvas_width, CV_8UC3 , m_color);
        //m_canvas = cv::Mat::zeros( canvas_height, canvas_width, CV_8UC3 );
        //tmp.copyTo(m_canvas);
        //cv::Mat cloneroi = m_canvas(cv::Rect(100, 10, 20,20));
        //cv::addWeighted(cloneroi,1.0, m_canvas, 0.3,0,cloneroi); 
        //tmp.copyTo(cloneroi);

        //int vert_step = (m_names.size() + 1) * box_height;
        int process_area_x = 0;
        int step = 0;
        std::string strname;
        for( int i = 0; i < m_names.size(); ++i ) {
            int box_y = box_height * i;
            int name_area_x = 0;
            strname = m_names[i];
            if(m_names[i].length() < max_string_length)
            {
                std::string strtmp(max_string_length, ' ');
                strname = strtmp.substr(0, max_string_length - strname.length()) + strname; 
            }
            strname += ":";

            cv::Size textsize = cv::getTextSize(strname, 0, m_font_scale, 0, NULL);

            //cv::putText( m_canvas, strname, cv::Point( name_area_x + letter_width, box_y + 2 * letter_height ), 0, m_font_scale, CV_RGB( 0, 255, 0 ) );

            process_area_x = name_area_x + name_area_width;
            //int process_area_x = name_area_x + name_area_width;

            cv::putText( m_canvas, strname, cv::Point( process_area_x - textsize.width - letter_width, box_y + 2 * letter_height ), 0, m_font_scale, CV_RGB( 0, 0, 255 ) );
            std::string percent_str = "LOW";
            float percent_f = 0;
            if(m_values[i] == 1)
            {
                percent_str = "MEDIUM";
                percent_f = 0.5;
            }else if(m_values[i] == 2)
            {
                percent_str = "HIGH";
                percent_f = 1;
            }


            if(m_values[i] == 0)
            {
                cv::putText( m_canvas, "LOW", cv::Point( process_area_x, box_y + 2 * letter_height ), 0, m_font_scale, 
                         CV_RGB( 255, 0, 0 ) );
                fillRectangle( m_canvas, cv::Rect( process_area_x + 4 * letter_width, box_y + 3, box_height - 6, box_height - 6 ), 
                           CV_RGB( 255, 0, 0 ), 1 );
            }else
            {
                cv::putText( m_canvas, "LOW", cv::Point( process_area_x, box_y + 2 * letter_height ), 0, m_font_scale, 
                         CV_RGB( 122, 122, 122 ) );
                cv::rectangle( m_canvas, cv::Rect( process_area_x + 4 * letter_width, box_y + 3, box_height - 6, box_height - 6 ), 
                            CV_RGB( 0, 0, 0 ), 1 );
                fillRectangle( m_canvas, cv::Rect( process_area_x + 4 * letter_width, box_y + 3, box_height - 6, box_height - 6 ), 
                           CV_RGB( 122, 122, 122 ), 1 );
                cv::rectangle( m_canvas, cv::Rect( process_area_x + 4 * letter_width + 3, box_y + 6, box_height - 12, box_height - 12 ), 
                           CV_RGB( 0, 0, 0 ), 1 );
            }
            step = 4 * letter_width + box_height - 6 + letter_width;

            if(m_values[i] == 1)
            {

                cv::putText( m_canvas, "MEDIUM", cv::Point( process_area_x + step, box_y + 2 * letter_height ), 0, m_font_scale, 
                         CV_RGB( 255, 255, 0 ) );
                fillRectangle( m_canvas, cv::Rect( process_area_x + 7 * letter_width + step, box_y + 3, box_height - 6, box_height - 6 ), 
                           CV_RGB(255 , 255, 0 ), 1 );
            }else
            { 
                cv::putText( m_canvas, "MEDIUM", cv::Point( process_area_x + step, box_y + 2 * letter_height ), 0, m_font_scale, 
                         CV_RGB( 122, 122, 122 ) );
                cv::rectangle( m_canvas, cv::Rect( process_area_x + 7 * letter_width + step, box_y + 3, box_height - 6, box_height - 6 ), 
                           CV_RGB( 0, 0, 0 ), 1 );
                fillRectangle( m_canvas, cv::Rect( process_area_x + 7 * letter_width + step, box_y + 3, box_height - 6, box_height - 6 ), 
                           CV_RGB( 122, 122, 122 ), 1 );
                cv::rectangle( m_canvas, cv::Rect( process_area_x + 7 * letter_width + step + 3, box_y + 6, box_height - 12, box_height - 12 ), 
                           CV_RGB( 0, 0, 0 ), 1 );
            }

            step += 7 * letter_width + box_height - 6 + letter_width;

            if(m_values[i] == 2)
            {
                cv::putText( m_canvas, "HIGH", cv::Point( process_area_x + step, box_y + 2 * letter_height ), 0, m_font_scale, 
                         CV_RGB( 0, 255, 0 ) );
                fillRectangle( m_canvas, cv::Rect( process_area_x + 5 * letter_width + step, box_y + 3, box_height - 6, box_height - 6 ), 
                           CV_RGB( 0, 255, 0 ), 1 );
            }else
            {

                cv::putText( m_canvas, "HIGH", cv::Point( process_area_x + step, box_y + 2 * letter_height ), 0, m_font_scale, 
                         CV_RGB( 122, 122, 122 ) );
                cv::rectangle( m_canvas, cv::Rect( process_area_x + 5 * letter_width + step, box_y + 3, box_height - 6, box_height - 6 ), 
                           CV_RGB( 0, 0, 0 ), 1 );
                fillRectangle( m_canvas, cv::Rect( process_area_x + 5 * letter_width + step, box_y + 3, box_height - 6, box_height - 6 ), 
                           CV_RGB( 122, 122, 122 ), 1 );
                cv::rectangle( m_canvas, cv::Rect( process_area_x + 5 * letter_width + step + 3, box_y + 6, box_height - 12, box_height - 12 ), 
                           CV_RGB( 0, 0, 0 ), 1 );
            }

        }


        ///////////////////////////
        //vert_step += box_height;

        int vert_step = (m_names.size() + 1) * box_height;
        strname = m_lbn;
        if(strname.length() < max_string_length)
        {
            std::string strtmp(max_string_length, ' ');
            strname = strtmp.substr(0, max_string_length - strname.length()) + strname;
        }
        strname += ":";

        step = 0;
        //step = 4 * letter_width + box_height - 6 + letter_width;
        cv::Size textsize = cv::getTextSize(strname, 0, m_font_scale, 0, NULL);
        cv::putText( m_canvas, strname, cv::Point( name_area_width - textsize.width - letter_width, vert_step + 2 * letter_height ), 0, m_font_scale, CV_RGB( 0, 0, 255 ) );
        if(m_light_value == 0)
        {
                cv::putText( m_canvas, "BRIGHT", cv::Point( process_area_x + step, vert_step + 2 * letter_height ), 0, m_font_scale,
                         CV_RGB( 0, 255, 0 ) );
                fillRectangle( m_canvas, cv::Rect( process_area_x + 6 * letter_width + step, vert_step + 3, box_height - 6, box_height - 6 ),
                           CV_RGB(0 , 255, 0 ), 1 );
        }else
        {
                cv::putText( m_canvas, "BRIGHT", cv::Point( process_area_x + step, vert_step + 2 * letter_height ), 0, m_font_scale,
                         CV_RGB( 122, 122, 122 ) );
                cv::rectangle( m_canvas, cv::Rect( process_area_x + 6 * letter_width + step, vert_step + 3, box_height - 6, box_height - 6 ),
                           CV_RGB( 0, 0, 0 ), 1 );
                fillRectangle( m_canvas, cv::Rect( process_area_x + 6 * letter_width + step, vert_step + 3, box_height - 6, box_height - 6 ),
                           CV_RGB( 122, 122, 122 ), 1 );
                cv::rectangle( m_canvas, cv::Rect( process_area_x + 6 * letter_width + step + 3, vert_step + 6, box_height - 12, box_height - 12 ),
                           CV_RGB( 0, 0, 0 ), 1 );
        }

        step += 6 * letter_width + box_height - 6 + letter_width;
        if(m_light_value == 1)
        {
                cv::putText( m_canvas, "    DARK", cv::Point( process_area_x + step, vert_step + 2 * letter_height ), 0, m_font_scale,
                         CV_RGB( 255, 0, 0 ) );
                fillRectangle( m_canvas, cv::Rect( process_area_x + 8 * letter_width + step, vert_step + 3, box_height - 6, box_height - 6 ),
                           CV_RGB(255 , 0, 0 ), 1 );
        }else
        {
                cv::putText( m_canvas, "    DARK", cv::Point( process_area_x + step, vert_step + 2 * letter_height ), 0, m_font_scale,
                         CV_RGB( 122, 122, 122 ) );
                cv::rectangle( m_canvas, cv::Rect( process_area_x + 8 * letter_width + step, vert_step + 3, box_height - 6, box_height - 6 ),
                           CV_RGB( 0, 0, 0 ), 1 );
                fillRectangle( m_canvas, cv::Rect( process_area_x + 8 * letter_width + step, vert_step + 3, box_height - 6, box_height - 6 ),
                           CV_RGB( 122, 122, 122 ), 1 );
                cv::rectangle( m_canvas, cv::Rect( process_area_x + 8 * letter_width + step + 3, vert_step + 6, box_height - 12, box_height - 12 ),
                           CV_RGB( 0, 0, 0 ), 1 );
        }

        /*
        step += 8 * letter_width + box_height - 6 + letter_width;
        if(m_light_value == 2)
        {
                cv::putText( m_canvas, "     NORMAL", cv::Point( process_area_x + step, vert_step + 2 * letter_height ), 0, m_font_scale,
                         CV_RGB( 0, 255, 0 ) );
                fillRectangle( m_canvas, cv::Rect( process_area_x + 11 * letter_width + step, vert_step + 3, box_height - 6, box_height - 6 ),
                           CV_RGB(0 , 255, 0 ), 1 );
        }else
        {
                cv::putText( m_canvas, "     NORMAL", cv::Point( process_area_x + step, vert_step + 2 * letter_height ), 0, m_font_scale,
                         CV_RGB( 122, 122, 122 ) );
                cv::rectangle( m_canvas, cv::Rect( process_area_x + 11 * letter_width + step, vert_step + 3, box_height - 6, box_height - 6 ),
                           CV_RGB( 0, 0, 0 ), 1 );
                fillRectangle( m_canvas, cv::Rect( process_area_x + 11 * letter_width + step, vert_step + 3, box_height - 6, box_height - 6 ),
                           CV_RGB( 122, 122, 122 ), 1 );
                cv::rectangle( m_canvas, cv::Rect( process_area_x + 11 * letter_width + step + 3, vert_step + 6, box_height - 12, box_height - 12 ),
                           CV_RGB( 0, 0, 0 ), 1 );
        }
        */
 
        vert_step += box_height;
        step = 0;
        if(m_blur_value == 0)
        {
                cv::putText( m_canvas, " CLEAR", cv::Point( process_area_x + step, vert_step + 2 * letter_height ), 0, m_font_scale,
                         CV_RGB( 0, 255, 0 ) );
                fillRectangle( m_canvas, cv::Rect( process_area_x + 6 * letter_width + step, vert_step + 3, box_height - 6, box_height - 6 ),
                           CV_RGB(0 , 255, 0 ), 1 );
        }else
        {
                cv::putText( m_canvas, " CLEAR", cv::Point( process_area_x + step, vert_step + 2 * letter_height ), 0, m_font_scale,
                         CV_RGB( 122, 122, 122 ) );
                cv::rectangle( m_canvas, cv::Rect( process_area_x + 6 * letter_width + step, vert_step + 3, box_height - 6, box_height - 6 ),
                           CV_RGB( 0, 0, 0 ), 1 );
                fillRectangle( m_canvas, cv::Rect( process_area_x + 6 * letter_width + step, vert_step + 3, box_height - 6, box_height - 6 ),
                           CV_RGB( 122, 122, 122 ), 1 );
                cv::rectangle( m_canvas, cv::Rect( process_area_x + 6 * letter_width + step + 3, vert_step + 6, box_height - 12, box_height - 12 ),
                           CV_RGB( 0, 0, 0 ), 1 );
        }

        step += 6 * letter_width + box_height - 6 + letter_width;
        if(m_blur_value == 1)
        {
                cv::putText( m_canvas, "    BLUR", cv::Point( process_area_x + step, vert_step + 2 * letter_height ), 0, m_font_scale,
                         CV_RGB( 255, 255, 0 ) );
                fillRectangle( m_canvas, cv::Rect( process_area_x + 8 * letter_width + step, vert_step + 3, box_height - 6, box_height - 6 ),
                           CV_RGB(255 , 255, 0 ), 1 );
        }else
        {
                cv::putText( m_canvas, "    BLUR", cv::Point( process_area_x + step, vert_step + 2 * letter_height ), 0, m_font_scale,
                         CV_RGB( 122, 122, 122 ) );
                cv::rectangle( m_canvas, cv::Rect( process_area_x + 8 * letter_width + step, vert_step + 3, box_height - 6, box_height - 6 ),
                           CV_RGB( 0, 0, 0 ), 1 );
                fillRectangle( m_canvas, cv::Rect( process_area_x + 8 * letter_width + step, vert_step + 3, box_height - 6, box_height - 6 ),
                           CV_RGB( 122, 122, 122 ), 1 );
                cv::rectangle( m_canvas, cv::Rect( process_area_x + 8 * letter_width + step + 3, vert_step + 6, box_height - 12, box_height - 12 ),
                           CV_RGB( 0, 0, 0 ), 1 );
        }

        /*
        step += 8 * letter_width + box_height - 6 + letter_width;
        if(m_blur_value == 2)
        {
                cv::putText( m_canvas, "SERIOUSBLUR", cv::Point( process_area_x + step, vert_step + 2 * letter_height ), 0, m_font_scale,
                         CV_RGB( 255, 0, 0 ) );
                fillRectangle( m_canvas, cv::Rect( process_area_x + 11 * letter_width + step, vert_step + 3, box_height - 6, box_height - 6 ),
                           CV_RGB(255 , 0, 0 ), 1 );
        }else
        {
                cv::putText( m_canvas, "SERIOUSBLUR", cv::Point( process_area_x + step, vert_step + 2 * letter_height ), 0, m_font_scale,
                         CV_RGB( 122, 122, 122 ) );
                cv::rectangle( m_canvas, cv::Rect( process_area_x + 11 * letter_width + step, vert_step + 3, box_height - 6, box_height - 6 ),
                           CV_RGB( 0, 0, 0 ), 1 );
                fillRectangle( m_canvas, cv::Rect( process_area_x + 11 * letter_width + step, vert_step + 3, box_height - 6, box_height - 6 ),
                           CV_RGB( 122, 122, 122 ), 1 );
                cv::rectangle( m_canvas, cv::Rect( process_area_x + 11 * letter_width + step + 3, vert_step + 6, box_height - 12, box_height - 12 ),
                           CV_RGB( 0, 0, 0 ), 1 );
        }
        */

        vert_step += box_height;
        step = 0;
        if(m_noise_value == 0)
        {
                cv::putText( m_canvas, " NOISE", cv::Point( process_area_x + step, vert_step + 2 * letter_height ), 0, m_font_scale,
                         CV_RGB( 255, 0, 0 ) );
                fillRectangle( m_canvas, cv::Rect( process_area_x + 6 * letter_width + step, vert_step + 3, box_height - 6, box_height - 6 ),
                           CV_RGB(255 , 0, 0 ), 1 );
        }else
        {
                cv::putText( m_canvas, " NOISE", cv::Point( process_area_x + step, vert_step + 2 * letter_height ), 0, m_font_scale,
                         CV_RGB( 122, 122, 122 ) );
                cv::rectangle( m_canvas, cv::Rect( process_area_x + 6 * letter_width + step, vert_step + 3, box_height - 6, box_height - 6 ),
                           CV_RGB( 0, 0, 0 ), 1 );
                fillRectangle( m_canvas, cv::Rect( process_area_x + 6 * letter_width + step, vert_step + 3, box_height - 6, box_height - 6 ),
                           CV_RGB( 122, 122, 122 ), 1 );
                cv::rectangle( m_canvas, cv::Rect( process_area_x + 6 * letter_width + step + 3, vert_step + 6, box_height - 12, box_height - 12 ),
                           CV_RGB( 0, 0, 0 ), 1 );
        }

        step += 6 * letter_width + box_height - 6 + letter_width;
        if(m_noise_value == 1)
        {
                cv::putText( m_canvas, " NONOISE", cv::Point( process_area_x + step, vert_step + 2 * letter_height ), 0, m_font_scale,
                         CV_RGB( 0, 255, 0 ) );
                fillRectangle( m_canvas, cv::Rect( process_area_x + 8 * letter_width + step, vert_step + 3, box_height - 6, box_height - 6 ),
                           CV_RGB(0 , 255, 0 ), 1 );
        }else
        {
                cv::putText( m_canvas, " NONOISE", cv::Point( process_area_x + step, vert_step + 2 * letter_height ), 0, m_font_scale,
                         CV_RGB( 122, 122, 122 ) );
                cv::rectangle( m_canvas, cv::Rect( process_area_x + 8 * letter_width + step, vert_step + 3, box_height - 6, box_height - 6 ),
                           CV_RGB( 0, 0, 0 ), 1 );
                fillRectangle( m_canvas, cv::Rect( process_area_x + 8 * letter_width + step, vert_step + 3, box_height - 6, box_height - 6 ),
                           CV_RGB( 122, 122, 122 ), 1 );
                cv::rectangle( m_canvas, cv::Rect( process_area_x + 8 * letter_width + step + 3, vert_step + 6, box_height - 12, box_height - 12 ),
                           CV_RGB( 0, 0, 0 ), 1 );
        }

        vert_step += box_height * 2;
        ///////////////////
        strname = m_mask;
        if(strname.length() < max_string_length)
        {
            std::string strtmp(max_string_length, ' ');
            strname = strtmp.substr(0, max_string_length - strname.length()) + strname;
        }
        strname += ":";

        step = 0;
        textsize = cv::getTextSize(strname, 0, m_font_scale, 0, NULL);
        cv::putText( m_canvas, strname, cv::Point( name_area_width - textsize.width - letter_width, vert_step + 2 * letter_height ), 0, m_font_scale, CV_RGB( 0, 0, 255 ) );
        if(m_maskvalues[0] == 1)
        {
                cv::putText( m_canvas, "LEFT EYE", cv::Point( process_area_x + step, vert_step + 2 * letter_height ), 0, m_font_scale,
                         CV_RGB( 255, 0, 0 ) );
                fillRectangle( m_canvas, cv::Rect( process_area_x + 8 * letter_width + step, vert_step + 3, box_height - 6, box_height - 6 ),
                           CV_RGB(255 , 0, 0 ), 1 );
        }else
        {
                cv::putText( m_canvas, "LEFT EYE", cv::Point( process_area_x + step, vert_step + 2 * letter_height ), 0, m_font_scale,
                         CV_RGB( 122, 122, 122 ) );
                cv::rectangle( m_canvas, cv::Rect( process_area_x + 8 * letter_width + step, vert_step + 3, box_height - 6, box_height - 6 ),
                           CV_RGB( 0, 0, 0 ), 1 );
                fillRectangle( m_canvas, cv::Rect( process_area_x + 8 * letter_width + step, vert_step + 3, box_height - 6, box_height - 6 ),
                           CV_RGB( 122, 122, 122 ), 1 );
                cv::rectangle( m_canvas, cv::Rect( process_area_x + 8 * letter_width + step + 3, vert_step + 6, box_height - 12, box_height - 12 ),
                           CV_RGB( 0, 0, 0 ), 1 );
        }
 
        step += 9 * letter_width + box_height - 6 + letter_width;
        if(m_maskvalues[1] == 1)
        {
                cv::putText( m_canvas, "RIGHT EYE", cv::Point( process_area_x + step, vert_step + 2 * letter_height ), 0, m_font_scale,
                         CV_RGB( 255, 0, 0 ) );
                fillRectangle( m_canvas, cv::Rect( process_area_x + 9 * letter_width + step, vert_step + 3, box_height - 6, box_height - 6 ),
                           CV_RGB(255 , 0, 0 ), 1 );
        }else
        {
                cv::putText( m_canvas, "RIGHT EYE", cv::Point( process_area_x + step, vert_step + 2 * letter_height ), 0, m_font_scale,
                         CV_RGB( 122, 122, 122 ) );
                cv::rectangle( m_canvas, cv::Rect( process_area_x + 9 * letter_width + step, vert_step + 3, box_height - 6, box_height - 6 ),
                           CV_RGB( 0, 0, 0 ), 1 );
                fillRectangle( m_canvas, cv::Rect( process_area_x + 9 * letter_width + step, vert_step + 3, box_height - 6, box_height - 6 ),
                           CV_RGB( 122, 122, 122 ), 1 );
                cv::rectangle( m_canvas, cv::Rect( process_area_x + 9 * letter_width + step + 3, vert_step + 6, box_height - 12, box_height - 12 ),
                           CV_RGB( 0, 0, 0 ), 1 );
        }

        step += 9 * letter_width + box_height - 6 + letter_width;
        if(m_maskvalues[2] == 1)
        {
                cv::putText( m_canvas, "NOSE", cv::Point( process_area_x + step, vert_step + 2 * letter_height ), 0, m_font_scale,
                         CV_RGB( 255, 0, 0 ) );
                fillRectangle( m_canvas, cv::Rect( process_area_x + 5 * letter_width + step, vert_step + 3, box_height - 6, box_height - 6 ),
                           CV_RGB(255 , 0, 0 ), 1 );
        }else
        {
                cv::putText( m_canvas, "NOSE", cv::Point( process_area_x + step, vert_step + 2 * letter_height ), 0, m_font_scale,
                         CV_RGB( 122, 122, 122 ) );
                cv::rectangle( m_canvas, cv::Rect( process_area_x + 5 * letter_width + step, vert_step + 3, box_height - 6, box_height - 6 ),
                           CV_RGB( 0, 0, 0 ), 1 );
                fillRectangle( m_canvas, cv::Rect( process_area_x + 5 * letter_width + step, vert_step + 3, box_height - 6, box_height - 6 ),
                           CV_RGB( 122, 122, 122 ), 1 );
                cv::rectangle( m_canvas, cv::Rect( process_area_x + 5 * letter_width + step + 3, vert_step + 6, box_height - 12, box_height - 12 ),
                           CV_RGB( 0, 0, 0 ), 1 );
        }

        step += 6 * letter_width + box_height - 6 + letter_width;
        if(m_maskvalues[3] == 1)
        {
                cv::putText( m_canvas, "LEFT MOUTH CORNER", cv::Point( process_area_x + step, vert_step + 2 * letter_height ), 0, m_font_scale,
                         CV_RGB( 255, 0, 0 ) );
                fillRectangle( m_canvas, cv::Rect( process_area_x + 17 * letter_width + step, vert_step + 3, box_height - 6, box_height - 6 ),
                           CV_RGB(255 , 0, 0 ), 1 );
        }else
        {
                cv::putText( m_canvas, "LEFT MOUTH CORNER", cv::Point( process_area_x + step, vert_step + 2 * letter_height ), 0, m_font_scale,
                         CV_RGB( 122, 122, 122 ) );
                cv::rectangle( m_canvas, cv::Rect( process_area_x + 17 * letter_width + step, vert_step + 3, box_height - 6, box_height - 6 ),
                           CV_RGB( 0, 0, 0 ), 1 );
                fillRectangle( m_canvas, cv::Rect( process_area_x + 17 * letter_width + step, vert_step + 3, box_height - 6, box_height - 6 ),
                           CV_RGB( 122, 122, 122 ), 1 );
                cv::rectangle( m_canvas, cv::Rect( process_area_x + 17 * letter_width + step + 3, vert_step + 6, box_height - 12, box_height - 12 ),
                           CV_RGB( 0, 0, 0 ), 1 );
        }

        step += 17 * letter_width + box_height - 6 + letter_width;
        if(m_maskvalues[4] == 1)
        {
                cv::putText( m_canvas, "RIGHT MOUTH CORNER", cv::Point( process_area_x + step, vert_step + 2 * letter_height ), 0, m_font_scale,
                         CV_RGB( 255, 0, 0 ) );
                fillRectangle( m_canvas, cv::Rect( process_area_x + 18 * letter_width + step, vert_step + 3, box_height - 6, box_height - 6 ),
                           CV_RGB(255 , 0, 0 ), 1 );
        }else
        {
                cv::putText( m_canvas, "RIGHT MOUTH CORNER", cv::Point( process_area_x + step, vert_step + 2 * letter_height ), 0, m_font_scale,
                         CV_RGB( 122, 122, 122 ) );
                cv::rectangle( m_canvas, cv::Rect( process_area_x + 18 * letter_width + step, vert_step + 3, box_height - 6, box_height - 6 ),
                           CV_RGB( 0, 0, 0 ), 1 );
                fillRectangle( m_canvas, cv::Rect( process_area_x + 18 * letter_width + step, vert_step + 3, box_height - 6, box_height - 6 ),
                           CV_RGB( 122, 122, 122 ), 1 );
                cv::rectangle( m_canvas, cv::Rect( process_area_x + 18 * letter_width + step + 3, vert_step + 6, box_height - 12, box_height - 12 ),
                           CV_RGB( 0, 0, 0 ), 1 );
        }


    }

private:
    int m_max;
    float m_font_scale;
    std::vector<std::string> m_names;
    std::vector<int> m_values;
    cv::Mat m_canvas;
  
    std::string m_mask;
    std::vector<int>         m_maskvalues;

    std::string m_lbn;
    int         m_light_value;   
    int         m_blur_value;   
    int         m_noise_value;   
    cv::Scalar  m_color;
};

