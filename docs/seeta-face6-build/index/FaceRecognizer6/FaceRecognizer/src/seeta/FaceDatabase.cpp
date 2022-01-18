#include "seeta/FaceDatabase.h"
#include "seeta/FaceRecognizer.h"

#include <orz/utils/log.h>
#include <orz/mem/need.h>
#include <array>
#include <cmath>
#include <fstream>
#include <orz/sync/shotgun.h>
#include <map>
#include <orz/sync/canyon.h>
#include "Mutex.h"
#include <stack>
#include "seeta/common_alignment.h"

#define VER_HEAD(x) #x "."
#define VER_TAIL(x) #x
#define GENERATE_VER(seq) FUN_MAJOR seq
#define FUN_MAJOR(x) VER_HEAD(x) FUN_MINOR
#define FUN_MINOR(x) VER_HEAD(x) FUN_SINOR
#define FUN_SINOR(x) VER_TAIL(x)

#define LIBRARY_VERSION GENERATE_VER( \
	(SEETA_FACE_DETECTOR_MAJOR_VERSION) \
	(SEETA_FACE_DETECTOR_MINOR_VERSION) \
	(SEETA_FACE_DETECTOR_SINOR_VERSION))

#define LIBRARY_NAME "FaceDatabase"

#define LOG_HEAD LIBRARY_NAME "(" LIBRARY_VERSION "): "

namespace seeta
{
	namespace SEETA_FACE_RECOGNIZE_NAMESPACE_VERSION
	{
		class FaceDatabase::Implement
		{
		public:
            using self = Implement;

            Implement(const SeetaModelSetting &setting, int extraction_core_number, int comparation_core_number)
			{
				seeta::ModelSetting exciting = setting;
				auto models = exciting.get_model();
				if (models.size() != 1)
				{
                    orz::Log(orz::FATAL) << LOG_HEAD << "Must input 1 model." << orz::crash;
				}
                orz::Log(orz::STATUS) << LOG_HEAD << "Loading models...";

				std::string model_filename = models[0];

                m_cores.resize(extraction_core_number);
                for (auto &core : m_cores)
                {
                    core = std::make_shared<seeta::FaceRecognizer>(exciting);
                }
                m_main_core = m_cores[0];

                m_extraction_gun.reset(new orz::Shotgun(extraction_core_number));
                m_comparation_gun.reset(new orz::Shotgun(comparation_core_number > 1 ? comparation_core_number : 0));
			}

            seeta::FaceRecognizer &core() { return *m_main_core; }
            const seeta::FaceRecognizer &core() const { return *m_main_core; }

            size_t extraction_core_number() const { return m_extraction_gun->size(); }
            size_t comparation_core_number() const { return m_comparation_gun->size(); }

            orz::Cartridge *ExtractParallel(const SeetaImageData &image, const SeetaPointF *points, float *features) const
            {
                if (!points || !features) return nullptr;
                seeta::ImageData local_image = image;
                std::vector<SeetaPointF> local_points(points, points + 5);
                return m_extraction_gun->fire([this, local_image, local_points, features](int id)
                {
                    m_cores[id]->Extract(local_image, local_points.data(), features);
                });
            }

            orz::Cartridge *ExtractCroppedFaceParallel(const SeetaImageData &image, float *features) const
            {
                if (!features) return nullptr;
                seeta::ImageData local_image = image;
                return m_extraction_gun->fire([this, local_image, features](int id)
                {
                    m_cores[id]->ExtractCroppedFace(local_image, features);
                });
            }

            void JoinExtraction() const
            {
                m_extraction_gun->join();
            }

            bool Extract(const SeetaImageData &image, const SeetaPointF *points, float *features) const
            {
                auto cart_extraction = ExtractParallel(image, points, features);
                if (cart_extraction == nullptr) return false;
                cart_extraction->join();
                return true;
            }

            orz::Cartridge *CompareParallel(const float *features1, const float *features2, float *similarity) const
            {
                if (!features1 || !features2 || !similarity) return nullptr;
                return m_comparation_gun->fire([this, features1, features2, similarity](int)
                {
                    *similarity = m_main_core->CalculateSimilarity(features1, features2);
                });
            }

            void JoinComparation() const
            {
                m_comparation_gun->join();
            }

            int64_t Insert(const std::shared_ptr<float> &features) const
            {
                unique_write_lock<rwmutex> _locker(m_db_mutex);
                auto new_index = m_max_index++;
                m_db.insert(std::make_pair(new_index, features));
                return new_index;
            }

            void InsertParallel(const std::shared_ptr<float> &features, int64_t *index) const
            {
                auto local_features = features;
                m_insertion_queue([this, local_features, index]()
                {
                    *index = Insert(local_features);
                });
            }

            void JoinInsertion() const
            {
                m_insertion_queue.join();
            }

            int Delete(int64_t index)
            {
                unique_write_lock<rwmutex> _locker(m_db_mutex);
                return int(m_db.erase(index));
            }

            size_t Count() const
            {
                unique_read_lock<rwmutex> _locker(m_db_mutex);
                return m_db.size();
            }

            void Clear()
            {
                unique_write_lock<rwmutex> _locker(m_db_mutex);
                m_db.clear();
                m_max_index = 0;
            }

            orz::Cartridge *RegisterParallel(const SeetaImageData &image, const SeetaPointF *points, int64_t *index) const
            {
                if (!points || !index) return nullptr;
                seeta::ImageData local_image = image;
                std::vector<SeetaPointF> local_points(points, points + 5);
                return m_extraction_gun->fire([this, local_image, local_points, index](int id)
                {
                    std::shared_ptr<float> features(new float[m_cores[id]->GetExtractFeatureSize()], std::default_delete<float[]>());
                    bool succeed = m_cores[id]->Extract(local_image, local_points.data(), features.get());
                    if (!succeed)
                    {
                        *index = -1;
                        return;
                    }
                    InsertParallel(features, index);
                });
            }

            orz::Cartridge *RegisterCroppedFaceParallel(const SeetaImageData &image, int64_t *index) const
            {
                if (!index) return nullptr;
                seeta::ImageData local_image = image;
                return m_extraction_gun->fire([this, local_image, index](int id)
                {
                    std::shared_ptr<float> features(new float[m_cores[id]->GetExtractFeatureSize()], std::default_delete<float[]>());
                    bool succeed = m_cores[id]->ExtractCroppedFace(local_image, features.get());
                    if (!succeed)
                    {
                        *index = -1;
                        return;
                    }
                    InsertParallel(features, index);
                });
            }

            void JoinRegisteration() const
            {
                m_extraction_gun->join();
                JoinInsertion();
            }

            size_t QueryTop(const float *features, size_t N, int64_t* index, float* similarity) const
            {
                unique_read_lock<rwmutex> _read_locker(m_db_mutex);

                std::vector<std::pair<int64_t, float>> result(m_db.size());
                {
                    std::unique_lock<std::mutex> _locker(m_comparation_mutex);
                    size_t i = 0;
                    for (auto &line : m_db)
                    {
                        result[i].first = line.first;
                        CompareParallel(features, line.second.get(), &result[i].second);
                        i++;
                    }
                    JoinComparation();
                }

                std::partial_sort(result.begin(), result.begin() + N, result.end(), [](
                    const std::pair<int64_t, float> &a, const std::pair<int64_t, float> &b) -> bool
                {
                    return a.second > b.second;
                });
                const size_t top_n = std::min(N, result.size());
                for (size_t i = 0; i < top_n; ++i)
                {
                    index[i] = result[i].first;
                    similarity[i] = result[i].second;
                }
                return top_n;
            }

            class IndexWithSimilarity
            {
            public:
                IndexWithSimilarity() = default;
                IndexWithSimilarity(int64_t index, float similarity)
                    : index(index), similarity(similarity) {}

                int64_t index = -1;
                float similarity = 0;
            };

            static size_t SortAbove(IndexWithSimilarity *data, size_t N, float threshold)
            {
                if (N == 0) return 0;
                std::stack<std::pair<int64_t, int64_t>> sort_stack;
                sort_stack.push(std::make_pair(0, N - 1));
                int64_t left_bound = -1;
                int64_t right_bound = int64_t(N);
                while (!sort_stack.empty())
                {
                    const auto const_left = sort_stack.top().first;
                    const auto const_right = sort_stack.top().second;
                    sort_stack.pop();
                    // end case
                    if (const_right < const_left)
                    {
                        continue;
                    }
                    else if (const_right == const_left)
                    {
                        const auto bound = const_left;
                        if (data[bound].similarity >= threshold)
                        {
                            left_bound = bound;
                        }
                        else
                        {
                            right_bound = bound;
                        }
                        continue;
                    }
                    // sort part
                    auto left = const_left;
                    auto right = const_right;
                    const auto flag = data[left];
                    while (left < right)
                    {
                        while (left < right && data[right].similarity <= flag.similarity) --right;
                        data[left] = data[right];
                        while (left < right && data[left].similarity >= flag.similarity) ++left;
                        data[right] = data[left];
                    }
                    const auto bound = left;
                    data[bound] = flag;
                    if (flag.similarity >= threshold)
                    {
                        left_bound = bound;
                        sort_stack.push(std::make_pair(const_left, bound));
                        sort_stack.push(std::make_pair(bound + 1, const_right));
                    }
                    else
                    {
                        right_bound = bound;
                        sort_stack.push(std::make_pair(const_left, bound));
                    }
                }

                int64_t sorted_size = left_bound + 1;
                for (; sorted_size < right_bound; ++sorted_size)
                {
                    if (data[sorted_size].similarity < threshold) break;
                }

                return size_t(sorted_size);
            }

            size_t QueryAbove(const float *features, float threshold, size_t N, int64_t* index, float* similarity) const
            {
                unique_read_lock<rwmutex> _read_locker(m_db_mutex);

                std::vector<IndexWithSimilarity> result(m_db.size());
                {
                    std::unique_lock<std::mutex> _locker(m_comparation_mutex);
                    size_t i = 0;
                    for (auto &line : m_db)
                    {
                        result[i].index = line.first;
                        CompareParallel(features, line.second.get(), &result[i].similarity);
                        i++;
                    }
                    JoinComparation();
                }
                // sort all above threshold
                size_t sorted = SortAbove(result.data(), m_db.size(), threshold);
                const size_t top_n = std::min(N, sorted);
                for (size_t i = 0; i < top_n; ++i)
                {
                    index[i] = result[i].index;
                    similarity[i] = result[i].similarity;
                }
                return top_n;
            }

            template <typename T>
            static size_t Write(StreamWriter &writer, const T &value)
            {
                return writer.write(reinterpret_cast<const char *>(&value), sizeof(T));
            }

            template <typename T>
            static size_t Read(StreamReader &reader, T &value)
            {
                return reader.read(reinterpret_cast<char *>(&value), sizeof(T));
            }

            template <typename T>
            static size_t Write(StreamWriter &writer, const T *arr, size_t size)
            {
                return writer.write(reinterpret_cast<const char *>(arr), sizeof(T) * size);
            }

            template <typename T>
            static size_t Read(StreamReader &reader, T *arr, size_t size)
            {
                return reader.read(reinterpret_cast<char *>(arr), sizeof(T) * size);
            }

#define MAGIC_SERIAL 0x7726

            bool Save(StreamWriter &writer) const
            {
                unique_read_lock<rwmutex> _locker(m_db_mutex);
                const int flag = MAGIC_SERIAL;
                Write(writer, flag);

                const uint64_t num = m_db.size();
                const uint64_t dim = m_main_core->GetExtractFeatureSize();

                Write(writer, num);
                Write(writer, dim);

                for (auto &line : m_db)
                {
                    auto &index = line.first;
                    auto &features = line.second;
                    // do save
                    Write(writer, index);
                    Write(writer, features.get(), size_t(dim));
                }
                
                orz::Log(orz::STATUS) << LOG_HEAD << "Loaded " << num << " faces";

                return true;
            }

            bool Load(StreamReader &reader)
            {
                unique_write_lock<rwmutex> _locker(m_db_mutex);

                int flag;
                Read(reader, flag);
                if (flag != MAGIC_SERIAL) {
                    orz::Log(orz::ERROR) << LOG_HEAD << "Load terminated, unsupported file format";
                    return false;
                }

                uint64_t num;
                uint64_t dim;
                Read(reader, num);
                Read(reader, dim);

                if (m_main_core != nullptr)
                {
                    if (dim != uint64_t(m_main_core->GetExtractFeatureSize())) {
                        orz::Log(orz::ERROR) << LOG_HEAD << "Load terminated, mismatch feature size";
                        return false;
                    }
                }

                m_db.clear();
                m_max_index = -1;

                for (size_t i = 0; i < num; ++i)
                {
                    int64_t index;
                    std::shared_ptr<float> features(new float[size_t(dim)], std::default_delete<float[]>());

                    Read(reader, index);
                    Read(reader, features.get(), size_t(dim));

                    m_db.insert(std::make_pair(index, features));
                    m_max_index = std::max(m_max_index, index);
                }
                m_max_index++;

                orz::Log(orz::STATUS) << LOG_HEAD << "Loaded " << num << " faces";

                return true;
            }

            seeta::FaceRecognizer *ExtractionCore(int id = 0)
            {
                if (id < 0 || size_t(id) >= m_cores.size())
                {
                    return nullptr;
                }
                return m_cores[id].get();
            }

		private:
            std::shared_ptr<seeta::FaceRecognizer> m_main_core;
            std::vector<std::shared_ptr<seeta::FaceRecognizer>> m_cores;
            std::shared_ptr<orz::Shotgun> m_extraction_gun;
            std::shared_ptr<orz::Shotgun> m_comparation_gun;

            mutable std::map<int64_t, std::shared_ptr<float>> m_db; // saving face db
            mutable int64_t m_max_index = 0;    ///< next saving id 

            mutable rwmutex m_db_mutex;
            mutable std::mutex m_comparation_mutex;
            orz::Canyon m_insertion_queue;
		};
	}
}

seeta::FaceDatabase::FaceDatabase(const SeetaModelSetting &setting)
	: m_impl(new Implement(setting, 1, 1))
{
}

seeta::FaceDatabase::FaceDatabase(const SeetaModelSetting& setting, int extraction_core_number,
    int comparation_core_number)
    : m_impl(new Implement(setting,
    extraction_core_number > 1 ? extraction_core_number : 1,
    comparation_core_number > 1 ? comparation_core_number : 1))
{
}

seeta::FaceDatabase::~FaceDatabase()
{
    delete m_impl;
}

int seeta::FaceDatabase::SetLogLevel(int level)
{
	return orz::GlobalLogLevel(orz::LogLevel(level));
}

int seeta::FaceDatabase::GetCropFaceWidthV2()
{
    return 256;
}

int seeta::FaceDatabase::GetCropFaceHeightV2()
{
    return 256;
}

int seeta::FaceDatabase::GetCropFaceChannelsV2()
{
    return 3;
}

bool seeta::FaceDatabase::CropFaceV2(const SeetaImageData& image, const SeetaPointF* points, SeetaImageData& face)
{

    float mean_shape[10] = {
        89.3095f, 72.9025f,
        169.3095f, 72.9025f,
        127.8949f, 127.0441f,
        96.8796f, 184.8907f,
        159.1065f, 184.7601f,
    };
    float local_points[10];
    for (int i = 0; i < 5; ++i)
    {
        local_points[2 * i] = float(points[i].x);
        local_points[2 * i + 1] = float(points[i].y);
    }

    face_crop_core(image.data, image.width, image.height, image.channels, face.data, GetCropFaceWidthV2(), GetCropFaceHeightV2(), local_points, 5, mean_shape, 256, 256);

    return true;
}


float seeta::FaceDatabase::Compare(const SeetaImageData& image1, const SeetaPointF* points1,
    const SeetaImageData& image2, const SeetaPointF* points2) const
{
    auto feature_size = m_impl->core().GetExtractFeatureSize();
    std::unique_ptr<float[]> features(new float[2 * feature_size]);
    auto cart1 = m_impl->ExtractParallel(image1, points1, features.get());
    if (cart1 == nullptr) return 0;
    auto cart2 = m_impl->ExtractParallel(image2, points2, features.get() + feature_size);
    if (cart2 == nullptr) return 0;
    cart1->join();
    cart2->join();
    return m_impl->core().CalculateSimilarity(features.get(), features.get() + feature_size);
}

float seeta::FaceDatabase::CompareByCroppedFace(const SeetaImageData& cropped_face_image1,
    const SeetaImageData& cropped_face_image2) const
{
    auto feature_size = m_impl->core().GetExtractFeatureSize();
    std::unique_ptr<float[]> features(new float[2 * feature_size]);
    auto cart1 = m_impl->ExtractCroppedFaceParallel(cropped_face_image1, features.get());
    if (cart1 == nullptr) return 0;
    auto cart2 = m_impl->ExtractCroppedFaceParallel(cropped_face_image2, features.get() + feature_size);
    if (cart2 == nullptr) return 0;
    cart1->join();
    cart2->join();
    return m_impl->core().CalculateSimilarity(features.get(), features.get() + feature_size);
}

int64_t seeta::FaceDatabase::Register(const SeetaImageData& image, const SeetaPointF* points)
{
    auto feature_size = m_impl->core().GetExtractFeatureSize();
    std::shared_ptr<float> features(new float[feature_size], std::default_delete<float[]>());
    auto cart_extraction = m_impl->ExtractParallel(image, points, features.get());
    if (cart_extraction == nullptr) return -1;
    cart_extraction->join();
    int64_t index = m_impl->Insert(features);
    return index;
}

int64_t seeta::FaceDatabase::RegisterByCroppedFace(const SeetaImageData& cropped_face_image)
{
    auto feature_size = m_impl->core().GetExtractFeatureSize();
    std::shared_ptr<float> features(new float[feature_size], std::default_delete<float[]>());
    auto cart_extraction = m_impl->ExtractCroppedFaceParallel(cropped_face_image, features.get());
    if (cart_extraction == nullptr) return -1;
    cart_extraction->join();
    int64_t index = m_impl->Insert(features);
    return index;
}

int seeta::FaceDatabase::Delete(int64_t index)
{
    return m_impl->Delete(index);
}

void seeta::FaceDatabase::Clear()
{
    m_impl->Clear();
}

size_t seeta::FaceDatabase::Count() const
{
    return m_impl->Count();
}

int64_t seeta::FaceDatabase::Query(const SeetaImageData& image, const SeetaPointF* points, float* similarity) const
{
    int64_t index = -1;
    float local_similarity = 0;
    size_t top_n = QueryTop(image, points, 1, &index, &local_similarity);
    if (top_n < 1) return index;
    if (similarity != nullptr) *similarity = local_similarity;
    return index;
}

int64_t seeta::FaceDatabase::QueryByCroppedFace(const SeetaImageData& cropped_face_image, float* similarity) const
{
    int64_t index = -1;
    float local_similarity = 0;
    size_t top_n = QueryTopByCroppedFace(cropped_face_image, 1, &index, &local_similarity);
    if (top_n < 1) return index;
    if (similarity != nullptr) *similarity = local_similarity;
    return index;
}

size_t seeta::FaceDatabase::QueryTop(const SeetaImageData& image, const SeetaPointF* points, size_t N, int64_t* index,
    float* similarity) const
{
    if (!index || !similarity) return 0;
    this->Join();
    const auto count = this->Count();
    if (count == 0) return 0;
    const auto feature_size = m_impl->core().GetExtractFeatureSize();
    std::unique_ptr<float[]> features(new float[feature_size]);
    auto cart_extraction = m_impl->ExtractParallel(image, points, features.get());
    if (cart_extraction == nullptr) return 0;
    cart_extraction->join();
    return m_impl->QueryTop(features.get(), N, index, similarity);
}

size_t seeta::FaceDatabase::QueryTopByCroppedFace(const SeetaImageData& cropped_face_image, size_t N, int64_t* index,
    float* similarity) const
{
    if (!index || !similarity) return 0;
    this->Join();
    const auto count = this->Count();
    if (count == 0) return 0;
    const auto feature_size = m_impl->core().GetExtractFeatureSize();
    std::unique_ptr<float[]> features(new float[feature_size]);
    auto cart_extraction = m_impl->ExtractCroppedFaceParallel(cropped_face_image, features.get());
    if (cart_extraction == nullptr) return 0;
    cart_extraction->join();
    return m_impl->QueryTop(features.get(), N, index, similarity);
}

size_t seeta::FaceDatabase::QueryAbove(const SeetaImageData& image, const SeetaPointF* points, float threshold, size_t N,
    int64_t* index, float* similarity) const
{
    if (!index || !similarity) return 0;
    this->Join();
    const auto count = this->Count();
    if (count == 0) return 0;
    const auto feature_size = m_impl->core().GetExtractFeatureSize();
    std::unique_ptr<float[]> features(new float[feature_size]);
    auto cart_extraction = m_impl->ExtractParallel(image, points, features.get());
    if (cart_extraction == nullptr) return 0;
    cart_extraction->join();
    return m_impl->QueryAbove(features.get(), threshold, N, index, similarity);
}

size_t seeta::FaceDatabase::QueryAboveByCroppedFace(const SeetaImageData& cropped_face_image, float threshold, size_t N,
    int64_t* index, float* similarity) const
{
    if (!index || !similarity) return 0;
    this->Join();
    const auto count = this->Count();
    if (count == 0) return 0;
    const auto feature_size = m_impl->core().GetExtractFeatureSize();
    std::unique_ptr<float[]> features(new float[feature_size]);
    auto cart_extraction = m_impl->ExtractCroppedFaceParallel(cropped_face_image, features.get());
    if (cart_extraction == nullptr) return 0;
    cart_extraction->join();
    return m_impl->QueryAbove(features.get(), threshold, N, index, similarity);
}

void seeta::FaceDatabase::RegisterParallel(const SeetaImageData& image, const SeetaPointF* points, int64_t* index)
{
    auto cart_registeration = m_impl->RegisterParallel(image, points, index);
    (void)(cart_registeration);
}

void seeta::FaceDatabase::RegisterByCroppedFaceParallel(const SeetaImageData& cropped_face_image, int64_t* index)
{
    auto cart_registeration = m_impl->RegisterCroppedFaceParallel(cropped_face_image, index);
    (void)(cart_registeration);
}

void seeta::FaceDatabase::Join() const
{
    m_impl->JoinRegisteration();
}

bool seeta::FaceDatabase::Save(const char* path) const
{
    FileWriter ofile(path, FileWriter::Binary);
    if (!ofile.is_opened()) return false;
    return Save(ofile);
}

bool seeta::FaceDatabase::Load(const char* path)
{
    FileReader ifile(path, FileWriter::Binary);
    if (!ifile.is_opened()) return false;
    return Load(ifile);
}

bool seeta::FaceDatabase::Save(StreamWriter& writer) const
{
    return m_impl->Save(writer);
}

bool seeta::FaceDatabase::Load(StreamReader& reader)
{
    return m_impl->Load(reader);
}


seeta::FaceRecognizer *seeta::FaceDatabase::ExtractionCore(int i)
{
    return m_impl->ExtractionCore(i);
}


