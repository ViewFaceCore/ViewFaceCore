//
// Created by kier on 2019-07-24.
//

#include "seeta/QualityOfBrightness.h"
#include "seeta/ImageProcess.h"
#include <cmath>

#include <orz/utils/except.h>
#include <orz/utils/log.h>

namespace seeta {
    namespace {
        struct Data {
             /**
             * [0, bright_thresh0) and [bright_thresh3, ~) => LOW
             * [bright_thresh0, bright_thresh1) and [bright_thresh2, bright_thresh3) => MEDIUM
             * [bright_thresh1, bright_thresh2) => HIGH
             */
            float bright_thresh0 = 70.0f;
            float bright_thresh1 = 100.0f;
            float bright_thresh2 = 210.0f;
            float bright_thresh3 = 230.0f;

			float middle_thresh = (bright_thresh1 + bright_thresh2) / 2;//more near the middle, more score get
        };
    }
    QualityOfBrightness::QualityOfBrightness(float v0, float v1, float v2, float  v3) {
        m_data = new Data;
		Data* inner_data = reinterpret_cast<Data *>(m_data);
		inner_data->bright_thresh0 = v0;
		inner_data->bright_thresh1 = v1;
		inner_data->bright_thresh2 = v2;
		inner_data->bright_thresh3 = v3;
    }

	QualityOfBrightness::QualityOfBrightness() {
		m_data = new Data;
	}

    QualityOfBrightness::~QualityOfBrightness() {
        delete reinterpret_cast<Data *>(m_data);
    }

    double seeta_mean(const unsigned char* gray_data, int width, int height)
    {
        auto count = width * height;
        long sum = 0;
        for (int i = 0; i < count; ++i)
        {
            sum += gray_data[i];
        }
        return double(sum) / count;
    }

	double seeta_mean(const seeta::Image &gray)
	{
		if (gray.channels() > 1) {//make sure image is gray
			orz::Log(orz::FATAL) << "image channels num must be 1" << orz::crash;
		}

		auto count = gray.width() * gray.height();
		long sum = 0;
		for (int i = 0; i < count; ++i)
		{
			sum += gray.data()[i];
		}
		return double(sum) / count;
	}

	float get_bright_score(Data* inner_data, float bright) {
		float middle_thresh = inner_data->middle_thresh;
		float bright_score = 1.0 / (abs(bright - middle_thresh) + 1);

		return bright_score;
	}

	float grid_max_bright(const seeta::Image &img, int rows, int cols)
	{
		int row_height = img.height() / rows;
		int col_width = img.width() / cols;
		float bright_val = FLT_MIN;
		for (int y = 0; y < rows; ++y)
		{
			for (int x = 0; x < cols; ++x)
			{
				seeta::Image grid = seeta::crop(img, seeta::Rect(x * col_width, y * row_height, col_width, row_height));
				auto this_grid_val = seeta_mean(grid.data(), grid.width(), grid.height());
				if (this_grid_val > bright_val) bright_val = this_grid_val;
			}
		}
		return std::max<float>(bright_val, 0);
	}

    QualityResult QualityOfBrightness::check(const SeetaImageData &image, const SeetaRect &face, const SeetaPointF *points,
                                       const int32_t N) {
        
        seeta::Image img(image);
		seeta::Image face_image(face.width, face.height, image.channels);
		seeta::fill(img, seeta::Rect(face.x, face.y, face.width, face.width), face_image);

        auto gray = seeta::gray(img);
		float bright_value = grid_max_bright(gray, 3, 3);

        QualityLevel level;
		Data* inner_data = reinterpret_cast<Data *>(m_data);
        if(bright_value < inner_data->bright_thresh0 || bright_value >= inner_data->bright_thresh3)
        {
            level = QualityLevel::LOW;
        }
        else if((bright_value >= inner_data->bright_thresh0 && bright_value < inner_data->bright_thresh1) ||
                (bright_value >= inner_data->bright_thresh2 && bright_value < inner_data->bright_thresh3))
        {
            level = QualityLevel::MEDIUM;
        }
        else if(bright_value >= inner_data->bright_thresh1 && bright_value < inner_data->bright_thresh2)
        {
            level = QualityLevel::HIGH;
        }

		//return QualityResult(level, bright_value);
        return QualityResult(level, get_bright_score(inner_data, bright_value));
    }
}
