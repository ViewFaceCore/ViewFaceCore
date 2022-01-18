#pragma once

#include <HolidayForward.h>
#include <functional>
#include "DataHelper.h"
#include "ImageProcess.h"

#if SEETANET_MAJOR_VERSION * 10000 + SEETANET_MINOR_VERSION * 100 + SEETANET_SINOR_VERSION >= 705
#define SEETANET_HAS_SeetaModelResetInput
#endif

#if SEETANET_MAJOR_VERSION * 10000 + SEETANET_MINOR_VERSION * 100 + SEETANET_SINOR_VERSION >= 707
#define SEETANET_HAS_SeetaSetNumThreadsEx
#endif

class need
{
public:
	template <typename FUNC, typename... Args>
	need(FUNC func, Args &&... args)
	{
		task = std::bind(func, std::forward<Args>(args)...);
	}
	~need() { task(); }
private:
	std::function<void()> task;
};

class ForwardNet
{
public:
	enum Device
	{
		AUTO = SEETA_DEVICE_AUTO,	/**< 自动检测，会优先使用 GPU */
        CPU  = SEETA_DEVICE_CPU,	/**< 使用 CPU 计算 */
        GPU  = SEETA_DEVICE_GPU,	/**< 使用 GPU 计算 */
	};

    static void SetSingleCalculationThreads(int num)
    {
        SeetaSetNumThreads(num);
    }

    void SetNetForwadThreads(int num)
    {
#ifdef SEETANET_HAS_SeetaSetNumThreadsEx
        if (net) {
            SeetaSetNumThreadsEx(net, num);
        }
#endif
    }

    static Device FinalDevice(Device device)
    {
        SeetaCNN_DEVICE_TYPE type = HOLIDAY_CNN_CPU_DEVICE;
        switch (device)
        {
        case AUTO:
            type = SeetaDefaultDevice();
            break;
        case CPU:
            break;
        case GPU:
            if (SeetaDefaultDevice() == HOLIDAY_CNN_GPU_DEVICE) type = HOLIDAY_CNN_GPU_DEVICE;
            break;
        }
        switch (type)
        {
        case HOLIDAY_CNN_CPU_DEVICE:
            return CPU;
        case HOLIDAY_CNN_GPU_DEVICE:
            return GPU;
        default:
            return CPU;
        }
    }

	ForwardNet() {}

	ForwardNet(const std::string &model_path,
		int batch_size,
		int input_channels,
		int input_height,
		int input_width,
		const std::string &output_blob_name,
		Device device = AUTO, int id = 0)
	{
		LoadModel(model_path, batch_size, input_channels, input_height, input_width, output_blob_name, device, id);
	}

	ForwardNet(
		const void *model_buffer,
		size_t model_size,
		int batch_size,
		int input_channels,
		int input_height,
		int input_width,
		const std::string &output_blob_name,
		Device device = AUTO, int id = 0)
	{
		LoadModel(model_buffer, model_size, batch_size, input_channels, input_height, input_width, output_blob_name, device, id);
	}
	
	bool LoadModel(const std::string &model_path,
		int batch_size,
		int input_channels,
		int input_height,
		int input_width,
		const std::string &output_blob_name,
		Device device = AUTO, int id = 0)
	{
		char *buffer;
		int64_t length;
		if (SeetaReadAllContentFromFile(model_path.c_str(), &buffer, &length)) return false;
		need release_buffer(SeetaFreeBuffer, buffer);

		return LoadModel(buffer, size_t(length),
			batch_size, input_channels, input_height, input_width,
			output_blob_name,
			device, id);
	}

	bool LoadModel(
		const void *model_buffer,
		size_t model_size,
		int batch_size,
		int input_channels,
		int input_height,
		int input_width,
		const std::string &output_blob_name,
		Device device = AUTO, int id = 0,
        const seeta::Size &core_size = seeta::Size(-1, -1))
	{
		//char *buffer;
		//int64_t length;
		//if (SeetaReadAllContentFromFile(model_path.c_str(), &buffer, &length)) return false;
		//need release_buffer(SeetaFreeBuffer, buffer);

		const char *buffer = reinterpret_cast<const char *>(model_buffer);
		int64_t length = static_cast<int64_t>(model_size);

		SeetaCNN_Model *model;
		if (SeetaReadModelFromBuffer(buffer, size_t(length), &model)) return false;
		need release_model(SeetaReleaseModel, model);

#ifdef SEETANET_HAS_SeetaModelResetInput
        SeetaModelResetInput(model, core_size.width, core_size.height);
#endif

		SeetaCNN_DEVICE_TYPE type = HOLIDAY_CNN_CPU_DEVICE;
		switch (device)
		{
		case AUTO:
			type = SeetaDefaultDevice();
			break;
		case CPU:
			break;
		case GPU:
			if (SeetaDefaultDevice() == HOLIDAY_CNN_GPU_DEVICE) type = HOLIDAY_CNN_GPU_DEVICE;
			break;
		}
		SeetaCNN_Net *local_net;

		if (type == HOLIDAY_CNN_GPU_DEVICE)
		{
			if (SeetaCreateNetGPU(model, batch_size, id, &local_net)) return false;
		}
		else
		{
			if (SeetaCreateNet(model, batch_size, type, &local_net)) return false;
		}

		if (net) SeetaReleaseNet(net);
		net = local_net;

		this->batch_size = batch_size;
		this->input_channels = input_channels;
		this->input_height = input_height;
		this->input_width = input_width;
		this->output_blob_name = output_blob_name;

		return true;
	}

	~ForwardNet()
	{
		SeetaReleaseNet(net);
	}

	bool valid() const
	{
		return net != nullptr;
	}

	seeta::Blob<float> Forward(const seeta::Blob<float> &input) const
	{
		return Forward(input, output_blob_name);
	}

	seeta::Blob<float> Forward(const seeta::Blob<float> &input, const std::string &blob_name) const
	{
		if (!net) return seeta::Blob<float>();
		SeetaCNN_InputOutputData data;
		data.buffer_type = SEETACNN_NCHW_FLOAT;
		data.number = input.shape(0);
		data.channel = input.shape(1);
		data.height = input.shape(2);
		data.width = input.shape(3);
		data.data_point_float = const_cast<float *>(input.data());

		SeetaNetKeepBlob(net, blob_name.c_str());
		need KeepNoBlob(SeetaNetKeepNoBlob, net);

		if (SeetaRunNetFloat(net, 1, &data)) return seeta::Blob<float>();

		if (SeetaGetFeatureMap(net, blob_name.c_str(), &data)) return seeta::Blob<float>();

		seeta::Blob<float> blob(data.number, data.channel, data.height, data.width);
		blob.copy_from(data.data_point_float);

		return std::move(blob);
	}

	seeta::Blob<float> Forward(const seeta::Image &input) const
	{
		return Forward(input, output_blob_name);
	}

	seeta::Blob<float> Forward(const seeta::Image &input, const std::string &blob_name) const
	{
		if (!net) return seeta::Blob<float>();
		SeetaCNN_InputOutputData data;
		data.buffer_type = SEETACNN_BGR_IMGE_CHAR;
		data.number = input.shape(0);
		data.channel = input.shape(3);
		data.height = input.shape(1);
		data.width = input.shape(2);
		data.data_point_char = const_cast<unsigned char *>(input.data());

		SeetaNetKeepBlob(net, blob_name.c_str());
		need KeepNoBlob(SeetaNetKeepNoBlob, net);

		if (SeetaRunNetChar(net, 1, &data)) return seeta::Blob<float>();

		if (SeetaGetFeatureMap(net, blob_name.c_str(), &data)) return seeta::Blob<float>();

		seeta::Blob<float> blob(data.number, data.channel, data.height, data.width);
		blob.copy_from(data.data_point_float);

		return std::move(blob);
	}

	seeta::Blob<float> Forward(const std::vector<seeta::Image> &input) const
	{
		return Forward(input, output_blob_name);
	}

	seeta::Blob<float> Forward(const std::vector<seeta::Image> &input, const std::string &blob_name) const
	{
		if (input.empty()) return seeta::Blob<float>();
		if (!net) return seeta::Blob<float>();
		seeta::Blob<uint8_t> fused_input(int(input.size()), input[0].height(), input[0].width(), input[0].channels());
		for (int i = 0; i < fused_input.shape(0); ++i)
		{
			input[i].copy_to(&fused_input.data(i, 0, 0, 0));
		}
		SeetaCNN_InputOutputData data;
		data.buffer_type = SEETACNN_BGR_IMGE_CHAR;
		data.number = fused_input.shape(0);
		data.channel = fused_input.shape(3);
		data.height = fused_input.shape(1);
		data.width = fused_input.shape(2);
		data.data_point_char = const_cast<unsigned char *>(fused_input.data());

		SeetaNetKeepBlob(net, blob_name.c_str());
		need KeepNoBlob(SeetaNetKeepNoBlob, net);

		if (SeetaRunNetChar(net, 1, &data)) return seeta::Blob<float>();

		if (SeetaGetFeatureMap(net, blob_name.c_str(), &data)) return seeta::Blob<float>();

		seeta::Blob<float> blob(data.number, data.channel, data.height, data.width);
		blob.copy_from(data.data_point_float);

		return std::move(blob);
	}

	bool JustForwad(const seeta::Image &input, const std::vector<std::string> &blob_names) const
	{
		if (!net) return false;
		SeetaCNN_InputOutputData data;
		data.buffer_type = SEETACNN_BGR_IMGE_CHAR;
		data.number = input.shape(0);
		data.channel = input.shape(3);
		data.height = input.shape(1);
		data.width = input.shape(2);
		data.data_point_char = const_cast<unsigned char *>(input.data());

		for (auto &blob_name : blob_names)
			SeetaNetKeepBlob(net, blob_name.c_str());
		need KeepNoBlob(SeetaNetKeepNoBlob, net);

		if (SeetaRunNetChar(net, 1, &data)) return false;

		return true;
	}

	bool JustForwad(const seeta::Image &input) const
	{
		return JustForwad(input, {});
	}

	bool JustForwad(const seeta::Blob<float> &input, const std::vector<std::string> &blob_names) const
	{
		if (!net) return false;
		SeetaCNN_InputOutputData data;
		data.buffer_type = SEETACNN_NCHW_FLOAT;
		data.number = input.shape(0);
		data.channel = input.shape(1);
		data.height = input.shape(2);
		data.width = input.shape(3);
		data.data_point_float = const_cast<float *>(input.data());

		for (auto &blob_name : blob_names)
			SeetaNetKeepBlob(net, blob_name.c_str());
		need KeepNoBlob(SeetaNetKeepNoBlob, net);

		if (SeetaRunNetFloat(net, 1, &data)) return false;

		return true;
	}

	bool JustForwad(const seeta::Blob<float> &input) const
	{
		return JustForwad(input, {});
	}

	seeta::Blob<float> Get(const std::string &blob_name) const
	{
		if (!net) return seeta::Blob<float>();

		SeetaCNN_InputOutputData data;

		if (SeetaGetFeatureMap(net, blob_name.c_str(), &data)) return seeta::Blob<float>();

		seeta::Blob<float> blob(data.number, data.channel, data.height, data.width);
		blob.copy_from(data.data_point_float);

		return std::move(blob);
	}

	int GetInputChannels() const { return input_channels; }
	int GetInputHeight() const { return input_height; }
	int GetInputWidth() const { return input_width; }

private:
	SeetaCNN_Net *net = nullptr;
	int batch_size = 0;
	int input_channels = 0;
	int input_height = 0;
	int input_width = 0;
	std::string output_blob_name = "";
};