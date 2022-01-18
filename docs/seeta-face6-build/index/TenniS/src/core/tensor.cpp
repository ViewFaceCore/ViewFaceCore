//
// Created by kier on 2018/5/25.
//

#include <core/tensor.h>

#include <utility>

#include <utils/ctxmgr.h>
#include "core/tensor.h"
#include "utils/assert.h"

#include <numeric>
#include <mutex>

#include <core/device_context.h>
#include <runtime/runtime.h>
#include <runtime/workbench.h>

namespace ts {
    struct EmptyMemoryKeeper {
    public:
        static void *Allocator(int id, size_t new_size, void *mem, size_t mem_size) {
            if (new_size == 0 && mem == nullptr) return nullptr;
            void *new_mem = nullptr;
            if (new_size == 0) {
                std::free(mem);
                return nullptr;
            } else if (mem != nullptr) {
                if (mem_size) {
                    new_mem = std::realloc(mem, new_size);
                } else {
                    std::free(mem);
                    new_mem = std::malloc(new_size);
                }
            } else {
                new_mem = std::malloc(new_size);
            }
            if (new_mem == nullptr) throw OutOfMemoryException(MemoryDevice(CPU, id), new_size);
            return new_mem;
        }

        EmptyMemoryKeeper() {
            Tensor::Prototype proto(VOID, Shape());
            auto memory = std::make_shared<HardMemory>(CPU, Allocator, proto.type_bytes() * proto.count());
            _empty_memory = new SyncMemory(memory, true);
            object = new Smart<SyncMemory>(_empty_memory, EmptyDeleter);
        }

        ~EmptyMemoryKeeper() {
			delete object;
            delete _empty_memory;
        }

        static void EmptyDeleter(const SyncMemory *memory) {}

        Smart<SyncMemory> &get() {
            return *object;
        }
    private:
        SyncMemory *_empty_memory = nullptr;
        Smart<SyncMemory> *object = nullptr;
    };
    namespace {
        EmptyMemoryKeeper empty_memory_keeper;
    }

    static Smart<SyncMemory> empty_memory() {
        return empty_memory_keeper.get().weak();
    }

    static Smart<SyncMemory> empty_memory(const MemoryDevice &device) {
        auto on_cpu = empty_memory_keeper.get().weak();
        if (device == on_cpu->device()) return on_cpu;
        return on_cpu->view(device);
    }
    static bool is_empty(const Tensor::Prototype &proto) {
        return proto.dtype() == VOID && proto.dims() == 0;
    }

//    static bool is_empty(DTYPE dtype, const Shape &shape) {
//        return dtype == VOID && shape.size() == 0;
//    }

    Tensor::Tensor(MemoryController::shared controller, DTYPE dtype, const Shape &_shape)
            : Tensor(controller, Prototype(dtype, _shape)) {}

    Tensor::Tensor(SyncMemoryController::shared controller, DTYPE dtype, const Shape &_shape)
            : Tensor(controller, Prototype(dtype, _shape)) {}

    Tensor::Tensor(SyncMemoryController::shared controller, DTYPE dtype, const Shape &_shape, const MemoryDevice &device)
            : Tensor(controller, Prototype(dtype, _shape), device) {}

    Tensor::Tensor(const MemoryDevice &device, DTYPE dtype, const Shape &_shape)
            : Tensor(device, Prototype(dtype, _shape)) {}

    Tensor::Tensor(DTYPE dtype, const Shape &_shape)
            : Tensor(Prototype(dtype, _shape)) {}

    Tensor::Tensor(MemoryController::shared controller, const Tensor::Prototype &proto)
            : m_memory(is_empty(proto)
                       ? empty_memory()
                       : Smart<TensorMemory>(controller->alloc(static_cast<size_t>(proto.count() * proto.type_bytes()))))
            , m_proto(proto) {}

    Tensor::Tensor(SyncMemoryController::shared controller, const Tensor::Prototype &proto)
            : m_memory(is_empty(proto)
                       ? empty_memory()
                       : controller->alloc(static_cast<size_t>(proto.count() * proto.type_bytes())))
            , m_proto(proto) {}

    Tensor::Tensor(SyncMemoryController::shared controller, const Tensor::Prototype &proto, const MemoryDevice &device)
            : m_memory(is_empty(proto)
                       ? empty_memory(device)
                       : controller->alloc(device, static_cast<size_t>(proto.count() * proto.type_bytes())))
            , m_proto(proto) {}

    Tensor::Tensor(const MemoryDevice &device, const Tensor::Prototype &proto)
            : m_memory(is_empty(proto)
                       ? empty_memory(device)
                       : make_smart<TensorMemory>(device, static_cast<size_t>(proto.count() * proto.type_bytes())))
            , m_proto(proto) {}

    Tensor::Tensor(const Tensor::Prototype &proto)
            : m_memory(is_empty(proto)
                       ? empty_memory()
                       : Smart<TensorMemory>(static_cast<size_t>(proto.count() * proto.type_bytes())))
            , m_proto(proto) {}

    Tensor::Tensor(const Memory &memory, const Tensor::Prototype &proto)
            : m_memory(memory)
            , m_proto(proto) {}

    Tensor::Tensor(const SyncMemory &memory, const Tensor::Prototype &proto)
            : m_memory(memory)
            , m_proto(proto) {}

    Tensor::Tensor(const Smart<TensorMemory> &memory, const Tensor::Prototype &proto)
            : m_memory(memory)
            , m_proto(proto) {}

    Tensor Tensor::clone() const {
        std::shared_ptr<MemoryController> controller = std::make_shared<DynamicMemoryController>(this->device());
        return clone(controller);
    }

    Tensor Tensor::clone(MemoryController::shared controller) const {
        auto fields = this->unpack();
        for (auto &value : fields) {
            Tensor dolly(controller, value.m_proto);
            auto dst = dolly.m_memory->weak_memory();
            auto src = value.m_memory->weak_memory();
            memcpy(dst, src, size_t(value.m_proto.count() * value.m_proto.type_bytes()));
            value = dolly;
        }
        Tensor dolly;
        dolly.pack(fields);
        return std::move(dolly);
    }

    Tensor Tensor::clone(SyncMemoryController::shared controller) const {
        auto fields = this->unpack();
        for (auto &value : fields) {
            Tensor dolly(controller, value.m_proto);
            auto dst = dolly.m_memory->weak_memory();
            auto src = value.m_memory->weak_memory();
            memcpy(dst, src, size_t(value.m_proto.count() * value.m_proto.type_bytes()));
            value = dolly;
        }
        Tensor dolly;
        dolly.pack(fields);
        return std::move(dolly);
    }

    Tensor Tensor::clone(SyncMemoryController::shared controller, const MemoryDevice &device) const {
        auto fields = this->unpack();
        for (auto &value : fields) {
            Tensor dolly(controller, value.m_proto, device);
            auto dst = dolly.m_memory->weak_memory();
            auto src = value.m_memory->weak_memory();
            memcpy(dst, src, size_t(value.m_proto.count() * value.m_proto.type_bytes()));
            value = dolly;
        }
        Tensor dolly;
        dolly.pack(fields);
        return std::move(dolly);
    }

    Tensor::shared Tensor::clone_shared(MemoryController::shared controller) const {
        return std::make_shared<Tensor>(this->clone(std::move(controller)));
    }

    Tensor::shared Tensor::clone_shared() const {
        return std::make_shared<Tensor>(this->clone());
    }

    Tensor::shared Tensor::clone_shared(SyncMemoryController::shared controller) const {
        return std::make_shared<Tensor>(this->clone(std::move(controller)));
    }

    Tensor::shared Tensor::clone_shared(SyncMemoryController::shared controller, const MemoryDevice &device) const {
        return std::make_shared<Tensor>(this->clone(std::move(controller), device));
    }

    Tensor::Tensor()
            : Tensor(VOID, Shape()) {
    }

    bool Tensor::empty() const {
        return dtype() == VOID && dims() == 0;
    }

    Tensor Tensor::reshape(const Shape &shape) const {
        auto fixed_shape = shape;
        int64_t fixed_index = -1;
        for (size_t i = 0; i < fixed_shape.size(); ++i) {
            if (fixed_shape[i] < 0) {
                if (fixed_index >= 0) {
                    TS_LOG_ERROR << "Can not reshape " << to_string(this->sizes()) << " to " << to_string(shape)
                                 << eject;
                }
                fixed_shape[i] = -1;
                fixed_index = int64_t(i);
            }
        }
        if (fixed_index >= 0) {
            auto up = std::accumulate(this->sizes().begin(), this->sizes().end(), 1, std::multiplies<int>());
            auto down = std::accumulate(fixed_shape.begin(), fixed_shape.end(), 1, std::multiplies<int>());
            fixed_shape[fixed_index] = up / -down;
        }

        Prototype proto(this->dtype(), fixed_shape);
        if (proto.count() != this->count()) {
            TS_LOG_ERROR << "Can not reshape " << to_string(this->sizes()) << " to " << to_string(shape) << eject;
        }
        Tensor t = *this;
        t.m_proto = std::move(proto);
        return std::move(t);
    }

    void Tensor::pack(const std::vector<Tensor::self> &fields) {
        if (fields.empty()) {
            this->m_memory = make_smart<TensorMemory>();
            this->m_proto = Prototype();
            return;
        }
        this->m_memory = fields[0].m_memory;
        this->m_proto = fields[0].m_proto;
        if (fields.size() > 1) {
            this->m_fields = std::vector<self>(fields.begin() + 1, fields.end());
        } else {
            this->m_fields.clear();
        }
    }

    std::vector<Tensor::self> Tensor::unpack() const {
        std::vector<Tensor::self> fields(1);
        fields[0].m_memory = this->m_memory;
        fields[0].m_proto = this->m_proto;
        if (!this->m_fields.empty()) {
            fields.insert(fields.end(), this->m_fields.begin(), this->m_fields.end());
        }
        return std::move(fields);
    }

    Tensor Tensor::field(size_t offset) const {
        if (offset == 0) {
            return Tensor(m_memory, m_proto);
        }
        if (offset - 1 >= m_fields.size()) {
            TS_LOG_ERROR << "Tensor offset output range error. Access index " << offset << " in range("
                         << fields_count() << ")" << eject;
        }
        return m_fields.at(offset - 1);
    }

    Tensor Tensor::field(int offset) const {
        return field(size_t(offset >= 0 ? offset : int(fields_count()) + offset));
    }

    void Tensor::field(size_t offset, const Tensor::self &value) {
        if (offset == 0) {
            this->m_memory = value.m_memory;
            this->m_proto = value.m_proto;
            return;
        }
        if (offset - 1 >= m_fields.size()) {
            TS_LOG_ERROR << "Tensor offset output range error. Access index " << offset << " in range("
                         << fields_count() << ")" << eject;
        }
        m_fields.at(offset - 1) = value;
    }

    void Tensor::field(int offset, const Tensor::self &value) {
        field(size_t(offset >= 0 ? offset : int(fields_count()) + offset), value);
    }

    size_t Tensor::fields_count() const {
        return 1 + m_fields.size();
    }

    bool Tensor::packed() const {
        return !m_fields.empty();
    }

    void Tensor::refield(size_t size) {
        if (size == 0) {
            *this = self();
        } else {
            m_fields.resize(size - 1);
        }
    }

    static size_t serialize_prototype_memory(StreamWriter &stream,
                                             const Tensor::Prototype &proto, const Memory &memory) {
        size_t writen_size = 0;
        // 1. write prototype
        // 1.1 write dtype
        writen_size += binio::write<uint8_t>(stream, proto.dtype());
        // 1.2 write size
        writen_size += binio::write<uint32_t>(stream, uint32_t(proto.sizes().size()));
        for (auto &size : proto.sizes()) {
            writen_size += binio::write<uint32_t>(stream, uint32_t(size));
        }
        Memory cpu_memory;
        if (memory.device().type() == ts::CPU) {
            cpu_memory = memory;
        } else {
            cpu_memory = Memory(memory.size());
            memcpy(cpu_memory, memory);
        }
        // 2. write memory
        writen_size += binio::write<char>(stream, cpu_memory.data<char>(), size_t(proto.count()) * proto.type_bytes());
        return writen_size;
    }

    static size_t externalize_prototype_memory(StreamReader &stream,
                                               Tensor::Prototype &proto, Memory &memory) {
        std::unique_ptr<MemoryController> may_controller(new DynamicMemoryController(memory.device()));
        auto controller = may_controller.get();
        size_t read_size = 0;
        // 1. read prototype
        DTYPE dtype;
        Shape shape;
        // 1.1 read dtype
        uint8_t dtype_buffer;
        read_size += binio::read<uint8_t >(stream, dtype_buffer);
        dtype = DTYPE(dtype_buffer);
        TS_AUTO_CHECK(dtype >= 0);
        // 1.2 read sizes
        uint32_t size_buffer;
        read_size += binio::read<uint32_t>(stream, size_buffer);
        shape.resize(size_buffer);
        for (size_t i = 0; i < shape.size(); ++i) {
            read_size += binio::read<uint32_t>(stream, size_buffer);
            shape[i] = size_buffer;
        }
        // 1.x set proto
        proto = Tensor::Prototype(dtype, shape);

        // 2. read memory
        memory = controller->alloc(size_t(proto.count()) * proto.type_bytes());
        read_size += binio::read<char>(stream, memory.data<char>(), memory.size());
        return read_size;
    }

    size_t Tensor::serialize(StreamWriter &stream) const {
        size_t writen_size = 0;
        writen_size += binio::write<uint32_t>(stream, uint32_t(this->fields_count()));
        for (auto &tensor : this->unpack()) {
            auto cpu_memory = tensor.m_memory->view(MemoryDevice(CPU)).weak_memory();
            writen_size += serialize_prototype_memory(stream, tensor.m_proto, cpu_memory);
        }
        return writen_size;
    }

    size_t Tensor::externalize(StreamReader &stream) {
        size_t read_size = 0;
        uint32_t size_buffer;
        read_size += binio::read<uint32_t>(stream, size_buffer);
        std::vector<Tensor> fields(size_buffer);
        for (auto &tensor : fields) {
            Memory cpu_memory;
            read_size += externalize_prototype_memory(stream, tensor.m_proto, cpu_memory);
            tensor.m_memory = SyncMemory(cpu_memory);
        }
        this->pack(fields);
        return read_size;
    }

    Tensor::Tensor(Tensor::self &&other) TS_NOEXCEPT {
        this->operator=(std::move(other));
    }

    Tensor::self &Tensor::operator=(Tensor::self &&other) TS_NOEXCEPT {
        this->m_proto = std::move(other.m_proto);
        this->m_memory = std::move(other.m_memory);
        this->m_fields = std::move(other.m_fields);
        return *this;
    }

    Tensor Tensor::view(const MemoryDevice &device) const {
        Tensor view_tensor;
        // view_tensor.m_memory = TensorMemory(m_memory->sync(device), false);
        view_tensor.m_memory = m_memory->view(device);
        view_tensor.m_proto = m_proto;

        if (!m_fields.empty()) {
            std::vector<self> view_fields(m_fields.size());
            for (size_t i = 0; i < m_fields.size(); ++i) {
                view_fields[i] = m_fields.at(i).view(device);
            }

            view_tensor.m_fields = std::vector<self>(std::move(view_fields));
        }

        return view_tensor;
    }

    void Tensor::broadcast()  {
        auto fields_count = this->fields_count();
        for (size_t i = 0; i < fields_count; ++i) {
            this->field(i).m_memory->broadcast();
        }
    }

    Tensor Tensor::weak() const {
        Tensor weak_tensor;
        weak_tensor.m_memory = m_memory.weak();
        weak_tensor.m_proto = m_proto;

        if (!m_fields.empty()) {
            std::vector<self> weak_fields(m_fields.size());
            for (size_t i = 0; i < m_fields.size(); ++i) {
                weak_fields[i] = m_fields.at(i).weak();
            }

            weak_tensor.m_fields = std::vector<self>(std::move(weak_fields));
        }

        return weak_tensor;
    }

    Tensor Tensor::strong() const {
        Tensor strong_tensor;
        strong_tensor.m_memory = m_memory.strong();
        strong_tensor.m_proto = m_proto;

        if (!m_fields.empty()) {
            std::vector<self> strong_fields(m_fields.size());
            for (size_t i = 0; i < m_fields.size(); ++i) {
                strong_fields[i] = m_fields.at(i).strong();
            }

            strong_tensor.m_fields = std::vector<self>(std::move(strong_fields));
        }

        return strong_tensor;
    }

    bool Tensor::has_shape(const Shape &shape) const {
        auto &this_shape = this->sizes();
        if (this_shape.size() != shape.size()) return false;
        for (size_t i = 0; i < shape.size(); ++i) {
            if (shape[i] >= 0 && this_shape[i] != shape[i]) return false;
        }
        return true;
    }

    bool Tensor::has_empty_shape() const {
        return this->sizes().empty();
    }

    static inline std::vector<int> flatten_shape(const Tensor &x, int m_dim) {
        auto need_size = size_t(m_dim + 1);
        auto x_size = x.sizes().size();
        if (need_size < x_size) {
            auto &size = x.sizes();
            std::vector<int> shape(size.begin(), size.begin() + need_size);
            shape.back() = std::accumulate(size.begin() + m_dim, size.end(), 1, std::multiplies<int>());
            return std::move(shape);
        } else if (need_size > x_size) {
            std::vector<int> ones(need_size - x_size, 1);
            auto shape = x.sizes();
            shape.insert(shape.end(), ones.begin(), ones.end());
            return std::move(shape.std());
        } else {
            return x.sizes().std();
        }
    }

    Tensor Tensor::flatten(int dim) const {
        auto fixed_shape = flatten_shape(*this, dim < 0 ? 0 : dim);
        Prototype proto(this->dtype(), fixed_shape);
        Tensor t = *this;
        t.m_proto = proto;
        return t;
    }

#define FAIL_SIZE(n) (this_shape.size() != (n))
#define FAIL_ARG(i) (arg##i >= 0 && this_shape[i] != arg##i)

    bool Tensor::has_shape(int arg0) const {
        auto &this_shape = this->sizes();
        return !(FAIL_SIZE(1) || FAIL_ARG(0));
    }

    bool Tensor::has_shape(int arg0, int arg1) const {
        auto &this_shape = this->sizes();
        return !(FAIL_SIZE(2) || FAIL_ARG(0) || FAIL_ARG(1));
    }

    bool Tensor::has_shape(int arg0, int arg1, int arg2) const {
        auto &this_shape = this->sizes();
        return !(FAIL_SIZE(3) || FAIL_ARG(0) || FAIL_ARG(1) || FAIL_ARG(2));
    }

    bool Tensor::has_shape(int arg0, int arg1, int arg2, int arg3) const {
        auto &this_shape = this->sizes();
        return !(FAIL_SIZE(4) || FAIL_ARG(0) || FAIL_ARG(1) || FAIL_ARG(2) || FAIL_ARG(3));
    }

    bool Tensor::has_shape(int arg0, int arg1, int arg2, int arg3, int arg4) const {
        auto &this_shape = this->sizes();
        return !(FAIL_SIZE(5) || FAIL_ARG(0) || FAIL_ARG(1) || FAIL_ARG(2) || FAIL_ARG(3) || FAIL_ARG(4));
    }

    bool Tensor::has_shape(int arg0, int arg1, int arg2, int arg3, int arg4,
                           int arg5) const {
        auto &this_shape = this->sizes();
        return !(FAIL_SIZE(6) || FAIL_ARG(0) || FAIL_ARG(1) || FAIL_ARG(2) || FAIL_ARG(3) || FAIL_ARG(4) ||
                 FAIL_ARG(5));
    }

    bool Tensor::has_shape(int arg0, int arg1, int arg2, int arg3, int arg4,
                           int arg5, int arg6) const {
        auto &this_shape = this->sizes();
        return !(FAIL_SIZE(7) || FAIL_ARG(0) || FAIL_ARG(1) || FAIL_ARG(2) || FAIL_ARG(3) || FAIL_ARG(4) ||
                 FAIL_ARG(5) || FAIL_ARG(6));
    }

    bool Tensor::has_shape(int arg0, int arg1, int arg2, int arg3, int arg4,
                           int arg5, int arg6, int arg7) const {
        auto &this_shape = this->sizes();
        return !(FAIL_SIZE(8) || FAIL_ARG(0) || FAIL_ARG(1) || FAIL_ARG(2) || FAIL_ARG(3) || FAIL_ARG(4) ||
                 FAIL_ARG(5) || FAIL_ARG(6) || FAIL_ARG(7));
    }

    bool Tensor::has_shape(int arg0, int arg1, int arg2, int arg3, int arg4,
                           int arg5, int arg6, int arg7, int arg8) const {
        auto &this_shape = this->sizes();
        return !(FAIL_SIZE(9) || FAIL_ARG(0) || FAIL_ARG(1) || FAIL_ARG(2) || FAIL_ARG(3) || FAIL_ARG(4) ||
                 FAIL_ARG(5) || FAIL_ARG(6) || FAIL_ARG(7) || FAIL_ARG(8));
    }

    bool Tensor::has_shape(int arg0, int arg1, int arg2, int arg3, int arg4,
                           int arg5, int arg6, int arg7, int arg8, int arg9) const {
        auto &this_shape = this->sizes();
        return !(FAIL_SIZE(10) || FAIL_ARG(0) || FAIL_ARG(1) || FAIL_ARG(2) || FAIL_ARG(3) || FAIL_ARG(4) ||
                 FAIL_ARG(5) || FAIL_ARG(6) || FAIL_ARG(7) || FAIL_ARG(8) || FAIL_ARG(9));
    }

    Memory Tensor::weak_memory() const {
        return m_memory->weak_memory();
    }

    Tensor::Tensor(Tensor::InFlow in_flow, const Tensor::Prototype &proto, const MemoryDevice &device) {
        switch (in_flow) {
            case InFlow::HOST: {
                auto flow = ctx::of<Workbench>::ref().runtime().flow();
                if (flow) {
                    *this = Tensor(flow, proto, MemoryDevice(CPU));
                } else {
                    throw Exception(std::string("Not flow binding in context: <") + typeid(RuntimeContext).name() + ">");
                }
                break;
            }
            case InFlow::DEVICE: {
                auto flow = ctx::of<Workbench>::ref().runtime().flow();
                if (flow) {
                    *this = Tensor(flow, proto, device);
                } else {
                    throw Exception(std::string("Not flow binding in context: <") + typeid(RuntimeContext).name() + ">");
                }
                break;
            }
        }
    }

    Tensor::Tensor(Tensor::InFlow in_flow, const Tensor::Prototype &proto) {
        switch (in_flow) {
            case InFlow::HOST: {
                auto flow = ctx::of<Workbench>::ref().runtime().flow();
                if (flow) {
                    *this = Tensor(flow, proto, MemoryDevice(CPU));
                } else {
                    throw Exception(std::string("Not flow binding in context: <") + typeid(RuntimeContext).name() + ">");
                }
                break;
            }
            case InFlow::DEVICE: {
                auto flow = ctx::of<Workbench>::ref().runtime().flow();
                auto device = ctx::of<Workbench>::ref().device().memory_device;
                if (flow) {
                    *this = Tensor(flow, proto, device);
                } else {
                    throw Exception(std::string("Not flow binding in context: <") + typeid(RuntimeContext).name() + ">");
                }
                break;
            }
        }
    }

    Tensor Tensor::view(Tensor::InFlow in_flow) const {
        switch (in_flow) {
            case InFlow::HOST: {
                return view(MemoryDevice(CPU, 0));
            }
            case InFlow::DEVICE: {
                return view(ctx::of<Workbench>::ref().device().memory_device);
            }
        }
        return *this;
    }

    Tensor Tensor::slice(int i) {
        auto &sizes = this->sizes();
        auto width = std::accumulate(sizes.begin() + 1, sizes.end(), 1, std::multiplies<int>());
        width *= this->proto().type_bytes();
        Shape slice_shape(sizes.begin() + 1, sizes.end());
        Memory slice_memory(this->device(), this->data<char>() + i * width, width);
        return Tensor(slice_memory, Tensor::Prototype(this->dtype(), slice_shape));
    }

    Tensor Tensor::slice(int beg, int end) {
        TS_AUTO_CHECK(beg < end);
        auto &sizes = this->sizes();
        auto width = std::accumulate(sizes.begin() + 1, sizes.end(), 1, std::multiplies<int>());
        width *= this->proto().type_bytes();

        auto batch = end - beg;
        Shape slice_shape = sizes;
        slice_shape[0] = batch;
        Memory slice_memory(this->device(), this->data<char>() + beg * width, batch * width);
        return Tensor(slice_memory, Tensor::Prototype(this->dtype(), slice_shape));
    }

    Tensor Tensor::Pack(const std::vector<Tensor> &fields) {
        Tensor x;
        x.pack(fields);
        return x;
    }

#undef FAIL_ARG
#undef FAIL_SIZE

    TensorPrototype::supper TensorPrototype::field(size_t offset) const {
        if (offset == 0) {
            return self(dtype(), sizes());
        }
        if (offset - 1 >= m_fields.size()) {
            TS_LOG_ERROR << "Tensor offset output range error. Access index " << offset << " in range("
                         << fields_count() << ")" << eject;
        }
        return m_fields.at(offset - 1);
    }

    void TensorPrototype::field(size_t offset, const TensorPrototype::supper &value) {
        if (offset == 0) {
            supper::operator=(value);
            return;
        }
        if (offset - 1 >= m_fields.size()) {
            TS_LOG_ERROR << "Tensor offset output range error. Access index " << offset << " in range("
                         << fields_count() << ")" << eject;
        }
        m_fields.at(offset - 1) = value;
    }


    void TensorPrototype::refield(size_t size) {
        if (size == 0) {
            *this = self();
        } else {
            m_fields.resize(size - 1);
        }
    }

    void TensorPrototype::pack(const std::vector<TensorPrototype::supper> &fields) {
        if (fields.empty()) {
            *this = self();
            return;
        }
        supper::operator=(fields[0]);
        if (fields.size() > 1) {
            this->m_fields = std::vector<supper>(fields.begin() + 1, fields.end());
        } else {
            this->m_fields.clear();
        }
    }

    std::vector<TensorPrototype::supper> TensorPrototype::unpack() const {
        std::vector<supper> fields(1);
        fields[0] = *this;
        if (!this->m_fields.empty()) {
            fields.insert(fields.end(), this->m_fields.begin(), this->m_fields.end());
        }
        return std::move(fields);
    }

    size_t TensorPrototype::fields_count() const {
        return 1 + m_fields.size();
    }

    bool TensorPrototype::packed() const {
        return !m_fields.empty();
    }

    TensorPrototype::TensorPrototype(const Tensor &tensor) {
        auto count = tensor.fields_count();
        m_fields.resize(count - 1);
        for (decltype(count) i = 0; i < count; ++i) {
            field(i, tensor.field(i).proto());
        }
    }

    TensorPrototype::TensorPrototype(const std::vector<Tensor::Prototype> &fields)
        : supper() {
        this->pack(fields);
    }

    TensorPrototype::supper TensorPrototype::field(int offset) const {
        return field(size_t(offset >= 0 ? offset : int(fields_count()) + offset));
    }

    void TensorPrototype::field(int offset, const TensorPrototype::supper &value) {
        return field(size_t(offset >= 0 ? offset : int(fields_count()) + offset), value);
    }

    static std::string to_checked_string(const Shape &shape) {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i) oss << ", ";
            oss << (shape[i] >= 0 ? std::to_string(shape[i]) : "?");
        }
        oss << "]";
        return oss.str();
    }

    std::ostream &operator<<(std::ostream &out, const Tensor::Prototype &proto) {
        std::ostringstream oss;
        oss << type_str(proto.dtype()) << ":" << to_checked_string(proto.sizes());
        return out << oss.str();
    }

    std::ostream &operator<<(std::ostream &out, const TensorPrototype &proto) {
        std::ostringstream oss;
        auto count = proto.fields_count();
        oss << "{";
        for (decltype(count) i = 0; i < count; ++i) {
            if (i) oss << ", ";
            oss << proto.field(i);
        }
        oss << "}";
        return out << oss.str();
    }

    bool operator==(const Tensor::Prototype &lhs, const Tensor::Prototype &rhs) {
        return lhs.dtype() == rhs.dtype() && lhs.sizes() == rhs.sizes();
    }

    bool operator==(const TensorPrototype &lhs, const TensorPrototype &rhs) {
        if (lhs.fields_count() != rhs.fields_count()) return false;
        auto count = lhs.fields_count();
        for (decltype(count) i = 0; i < count; ++i) {
            if (lhs.field(i) != rhs.field(i)) return false;
        }
        return true;
    }

    bool operator==(const Tensor::Prototype &lhs, const TensorPrototype &rhs) {
        return rhs.fields_count() == 1 && lhs.dtype() == rhs.dtype() && lhs.sizes() == rhs.sizes();
    }

    bool operator==(const TensorPrototype &lhs, const Tensor::Prototype &rhs) {
        return lhs.fields_count() == 1 && lhs.dtype() == rhs.dtype() && lhs.sizes() == rhs.sizes();
    }
}