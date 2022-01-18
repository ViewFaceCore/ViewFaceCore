//
// Created by kevin on 2019-08-29
//
#include "seeta/QualityOfClarity.h"
#include "seeta/ImageProcess.h"
namespace seeta{
	namespace {
		struct Data {
			/**
			* [0, low_thresh) => LOW
			* [low_thresh, high_thresh) => MEDIUM
			* [high_thresh, ~) => HIGH
			*/
			float low_thresh = 0.10f;
			float high_thresh = 0.20f;
		};
	}
        QualityOfClarity::QualityOfClarity(){
			m_data = new Data;
        }
		QualityOfClarity::QualityOfClarity(float low_thresh, float high_thresh) {
			m_data = new Data;

			Data* inner_data = reinterpret_cast<Data *>(m_data);
			inner_data->low_thresh = low_thresh;
			inner_data->high_thresh = high_thresh;
		}
		QualityOfClarity::~QualityOfClarity() {
			delete reinterpret_cast<Data *>(m_data);
		}

		float reblur(const unsigned char *data, int width, int height)
		{
			float blur_val = 0.0f;
			float kernel[9] = { 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f };
			float *BVer = new float[width * height];//垂直方向低通滤波后的结果
			float *BHor = new float[width * height];//水平方向低通滤波后的结果

			float filter_data = 0.0;
			for (int i = 0; i < height; ++i)//均值滤波
			{
				for (int j = 0; j < width; ++j)
				{
					if (i < 4 || i > height - 5)
					{//处理边界 直接赋值原数据
						BVer[i * width + j] = data[i * width + j];
					}
					else
					{
						filter_data = kernel[0] * data[(i - 4) * width + j] + kernel[1] * data[(i - 3) * width + j] + kernel[2] * data[(i - 2) * width + j] +
							kernel[3] * data[(i - 1) * width + j] + kernel[4] * data[(i)* width + j] + kernel[5] * data[(i + 1) * width + j] +
							kernel[6] * data[(i + 2) * width + j] + kernel[7] * data[(i + 3) * width + j] + kernel[8] * data[(i + 4) * width + j];
						BVer[i * width + j] = filter_data;
					}

					if (j < 4 || j > width - 5)
					{
						BHor[i * width + j] = data[i * width + j];
					}
					else
					{
						filter_data = kernel[0] * data[i * width + (j - 4)] + kernel[1] * data[i * width + (j - 3)] + kernel[2] * data[i * width + (j - 2)] +
							kernel[3] * data[i * width + (j - 1)] + kernel[4] * data[i * width + j] + kernel[5] * data[i * width + (j + 1)] +
							kernel[6] * data[i * width + (j + 2)] + kernel[7] * data[i * width + (j + 3)] + kernel[8] * data[i * width + (j + 4)];
						BHor[i * width + j] = filter_data;
					}

				}
			}

			float D_Fver = 0.0;
			float D_FHor = 0.0;
			float D_BVer = 0.0;
			float D_BHor = 0.0;
			float s_FVer = 0.0;//原始图像数据的垂直差分总和 对应论文中的 s_Fver
			float s_FHor = 0.0;//原始图像数据的水平差分总和 对应论文中的 s_Fhor
			float s_Vver = 0.0;//模糊图像数据的垂直差分总和 s_Vver
			float s_VHor = 0.0;//模糊图像数据的水平差分总和 s_VHor
			for (int i = 1; i < height; ++i)
			{
				for (int j = 1; j < width; ++j)
				{
					D_Fver = std::abs((float)data[i * width + j] - (float)data[(i - 1) * width + j]);
					s_FVer += D_Fver;
					D_BVer = std::abs((float)BVer[i * width + j] - (float)BVer[(i - 1) * width + j]);
					s_Vver += std::max((float)0.0, D_Fver - D_BVer);

					D_FHor = std::abs((float)data[i * width + j] - (float)data[i * width + (j - 1)]);
					s_FHor += D_FHor;
					D_BHor = std::abs((float)BHor[i * width + j] - (float)BHor[i * width + (j - 1)]);
					s_VHor += std::max((float)0.0, D_FHor - D_BHor);
				}
			}
			float b_FVer = (s_FVer - s_Vver) / s_FVer;
			float b_FHor = (s_FHor - s_VHor) / s_FHor;
			blur_val = std::max(b_FVer, b_FHor);

			delete[] BVer;
			delete[] BHor;

			return blur_val;
		}

		float grid_max_reblur(const seeta::Image &img, int rows, int cols)
		{
			int row_height = img.height() / rows;
			int col_width = img.width() / cols;
			float blur_val = FLT_MIN;
			for (int y = 0; y < rows; ++y)
			{
				for (int x = 0; x < cols; ++x)
				{
					seeta::Image grid = seeta::crop(img, seeta::Rect(x * col_width, y * row_height, col_width, row_height));
					auto this_grad_blur_val = reblur(grid.data(), grid.width(), grid.height());
					if (this_grad_blur_val > blur_val) blur_val = this_grad_blur_val;
				}
			}
			return std::max<float>(blur_val, 0);
		}


		float clarity_estimate(const SeetaImageData &image, const SeetaRect &info)
		{
			if (!image.data || info.width < 9 || info.height < 9) return 0.0;
			seeta::Image color_data(image.data, image.width, image.height, image.channels);
			seeta::Image gray_data = seeta::gray(color_data);

			seeta::Image src_data = seeta::crop(gray_data, seeta::Rect(info.x, info.y, info.width, info.height));
			// float blur_val = ReBlur(src_data.data(), src_data.width(), src_data.height());
			float blur_val = grid_max_reblur(src_data, 2, 2);
			float clarity = 1.0f - blur_val;

			float T1 = 0.0f;
			float T2 = 1.0f;
			if (clarity <= T1)
			{
				clarity = 0.0;
			}
			else if (clarity >= T2)
			{
				clarity = 1.0;
			}
			else
			{
				clarity = (clarity - T1) / (T2 - T1);
			}

			return clarity;
		}

		QualityResult QualityOfClarity::check(const SeetaImageData &image,
                                              const SeetaRect&face,
                                              const SeetaPointF* points,
                                              const int32_t num)
        {
			float clarity = clarity_estimate(image, face);

			Data* inner_data = reinterpret_cast<Data *>(m_data);

			QualityLevel level;
			if (clarity < inner_data->low_thresh)
			{
				level = QualityLevel::LOW;
			}
			else if (clarity >= inner_data->low_thresh && clarity < inner_data->high_thresh)
			{
				level = QualityLevel::MEDIUM;
			}
			else if (clarity >= inner_data->high_thresh)
			{
				level = QualityLevel::HIGH;
			}

			return QualityResult(level, clarity);
		}
 }

